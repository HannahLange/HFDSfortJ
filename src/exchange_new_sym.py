# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Any
from netket.utils.dispatch import dispatch
import jax
import numpy as np
from flax import struct
from jax import numpy as jnp
from netket.utils import struct

from netket.graph import AbstractGraph

from netket.sampler.rules.base import MetropolisRule
from netket import config

from netket.jax import apply_chunked
from functools import partial
from netket.jax.sharding import sharding_decorator

# Necessary for the type annotation to work
if config.netket_sphinx_build:
    from netket import sampler
from netket.sampler.rules import ExchangeRule
from netket.experimental.hilbert import SpinOrbitalFermions
from netket.graph import disjoint_union

class ParticleExchangeRule(ExchangeRule):
    """Exchange rule for particles on a lattice.

    Works similarly to :class:`netket.sampler.rules.ExchangeRule`, but
    takes into account that only occupied orbitals
    can be exchanged with unoccupied ones.

    This sampler conserves the number of particles.
    """

    def __init__(
        self,
	hilbert,
        *,
        clusters: Optional[list[tuple[int, int]]] = None,
        graph: Optional[AbstractGraph] = None,
        d_max: int = 1,
        exchange_spins: bool = False,
    ):
        r"""
        Constructs the ParticleExchange Rule.

        Particles are only exchanged between modes where the particle number is different.
        For fermions, only occupied orbitals can be exchanged with unoccupied ones.

        You can pass either a list of clusters or a netket graph object to
        determine the clusters to exchange.

        Args:
            hilbert: The hilbert space to be sampled.
            clusters: The list of clusters that can be exchanged. This should be
                a list of 2-tuples containing two integers. Every tuple is an edge,
                or cluster of sites to be exchanged.
            graph: A graph, from which the edges determine the clusters
                that can be exchanged.
            d_max: Only valid if a graph is passed in. The maximum distance
                between two sites
            exchange_spins: (default False) If exchange_spins, the graph must encode the
                connectivity  between the first N physical sites having same spin, and
                it is replicated using :func:`netket.graph.disjoint_union` other every
                spin subsector. This option conserves the number of fermions per
                spin subsector. If the graph does not have a number of sites equal
                to the number of orbitals in the hilbert space, this flag has no effect.
        """
        if not isinstance(hilbert, SpinOrbitalFermions):
            raise ValueError(
                "This sampler rule currently only works with SpinOrbitalFermions hilbert spaces."
            )
        if not exchange_spins and hilbert.n_spin_subsectors > 1:
            if graph is not None and graph.n_nodes == hilbert.n_orbitals:
                graph = disjoint_union(*[graph] * hilbert.n_spin_subsectors)
            if clusters is not None and np.max(clusters) < hilbert.n_orbitals:
                clusters = np.concatenate(
                    [
                        clusters + i * hilbert.n_orbitals
                        for i in range(hilbert.n_spin_subsectors)
                    ]
                )
        super().__init__(clusters=clusters, graph=graph, d_max=d_max)

    def random_state(self, sampler, machine, params, state, rng):
      apply_machine = apply_chunked(machine.apply, in_axes=(None, 0), chunk_size=4096)

      def loop_body(val):
          n, x, neg_inf_mask, params, rng, sampler = val
          n += 1
          probs = apply_machine(params, x).real
          neg_inf_mask = jnp.isneginf(probs) | jnp.isnan(probs) #| (jnp.abs(probs) >= 35) 
          num_trues = jnp.sum(neg_inf_mask)
          jax.debug.print("Number of inf values: {num}", num=(num_trues, jnp.sum(jnp.isnan(probs))))

          neg_inf_mask = jnp.repeat(jnp.expand_dims(neg_inf_mask, 1), x.shape[1], 1)
          rng = rng + sampler.n_batches*2
          #x_rep = sampler.hilbert.random_state(rng, size=sampler.n_batches, dtype=sampler.dtype)
          x_rep = self.random_state_(sampler, rng)
          x = jax.lax.select(neg_inf_mask, x_rep, x)
          return n, x, neg_inf_mask, params, rng, sampler

      def loop_cond(val):
          n, _, neg_inf_mask, _, _, _ = val
          neg_inf_mask = neg_inf_mask[:,0]
          return jnp.logical_and(jnp.any(neg_inf_mask), jnp.logical_or(jnp.sum(neg_inf_mask) > neg_inf_mask.shape[0]-1, n < 100))

      x = self.random_state_(sampler, rng)
      
      probs = apply_machine(params, x).real
      jax.debug.print("inital probs={z}",z=(probs, jnp.max(probs), jnp.min(probs)))
      neg_inf_mask = jnp.isneginf(probs) | jnp.isnan(probs) #| (jnp.abs(probs) >= 35) 
      neg_inf_mask = jnp.repeat(jnp.expand_dims(neg_inf_mask, 1), x.shape[1], 1)
      initial_val = (0, x, neg_inf_mask, params, rng, sampler)

      # Execute the while loop
      final_val = jax.lax.while_loop(loop_cond, loop_body, initial_val)
      _, x, neg_inf_mask, _, _, _ = final_val
      jax.debug.print('sum={x}', x=neg_inf_mask.shape)
      jax.debug.print('batches={x}', x=sampler.n_batches)

      # replace all -infs by first non -inf sample (loop through more rows if not converged)
      probs = apply_machine(params, x).real
      neg_inf_mask = jnp.isneginf(probs) | jnp.isnan(probs) #| (jnp.abs(probs) >= 35)
      sorted_indices = jnp.argsort(probs)
      jax.debug.print('sorted={x}', x=(neg_inf_mask[sorted_indices[-1]],probs[sorted_indices[-1]]))
      x_rep = x[sorted_indices[-1]]
      neg_inf_mask = jnp.repeat(jnp.expand_dims(neg_inf_mask, 1), x.shape[1], 1)
      x = jnp.where(neg_inf_mask, x_rep, x)      
          
      probs = apply_machine(params, x).real
      neg_inf_mask = jnp.isneginf(probs) | jnp.isnan(probs) #| (jnp.abs(probs) >= 35) 
      neg_inf_mask = jnp.repeat(jnp.expand_dims(neg_inf_mask, 1), x.shape[1], 1)
      num_trues = jnp.sum(neg_inf_mask)
      jax.debug.print("Number of inf values: {num}", num=(num_trues, jnp.sum(jnp.isnan(probs))))
      jax.debug.print("unique samples: {sam}", sam=jnp.sum(jnp.unique(x, fill_value=0, size=sampler.n_batches, axis=0))/sampler.hilbert.n_fermions)
      jax.debug.print("nans={x}", x=jnp.sum(jnp.isnan(probs)))
      jax.debug.print("final probs={z}",z=(probs, jnp.max(probs), jnp.min(probs)))
      
      return x.astype(jnp.float64)

    def random_state_(self, sampler, rng):
        #random sample with single occupancy
        batch_size = sampler.n_batches
        keys = jax.random.split(rng, batch_size)
        indices = jax.vmap(lambda key: jax.random.permutation(key, jnp.arange(sampler.hilbert.n_orbitals)))(keys)
        n_up = sampler.hilbert.n_fermions_per_spin[0]
        n_dn = sampler.hilbert.n_fermions_per_spin[1]
        up_ind = indices[:,:n_up]
        dn_ind = indices[:, n_up:(n_up+n_dn)]+sampler.hilbert.n_orbitals
        
       	states = jnp.zeros((batch_size, sampler.hilbert.size))
        rows = jnp.arange(states.shape[0])
        
        #insert up spins
        states = states.at[(rows, up_ind.T)].set(1)
        #insert down spins
        states = states.at[(rows, dn_ind.T)].set(1)
        return states

    def transition(rule, sampler, machine, parameters, state, key, σ):
        n_chains = σ.shape[0]

        # compute a mask for the clusters that can be hopped
        hoppable_clusters = _compute_hoppable_clusters_mask_f(rule.clusters, σ)

        keys = jnp.asarray(jax.random.split(key, n_chains))

        # we use shard_map to avoid the all-gather coming from the batched jnp.take / indexing
        @partial(sharding_decorator, sharded_args_tree=(True, True, True))
        @jax.vmap
        def _update_samples(key, σ, hoppable_clusters):
            # pick a random cluster, taking into account the mask
            n_conn = hoppable_clusters.sum(axis=-1)
            cluster = jax.random.choice(
                key,
                a=jnp.arange(rule.clusters.shape[0]),
                p=hoppable_clusters,
                replace=True,
            )

            # sites to be exchanged
            si = rule.clusters[cluster, 0]
            sj = rule.clusters[cluster, 1]

            σp = σ.at[si].set(σ[sj])
            σp = σp.at[sj].set(σ[si])

            # compute the number of connected sites
            hoppable_clusters_proposed = _compute_hoppable_clusters_mask_f(
                rule.clusters, σp
            )
            n_conn_proposed = hoppable_clusters_proposed.sum(axis=-1)
            log_prob_corr = jnp.log(n_conn) - jnp.log(n_conn_proposed)
            return σp, log_prob_corr

        return _update_samples(keys, σ, hoppable_clusters)

    def __repr__(self):
        return f"ParticleExchangeRule(# of clusters: {len(self.clusters)})"

@jax.jit
def _compute_hoppable_clusters_mask_f(clusters, σ):
    # mask the clusters to include only feasible moves (occ -> unocc, or the inverse)
    hoppable_clusters_mask = ~jnp.isclose(
        σ[..., clusters[:, 0]], σ[..., clusters[:, 1]]
    )
    return hoppable_clusters_mask




def compute_clusters(graph: AbstractGraph, d_max: int):
    """
    Given a netket graph and a maximum distance, computes all clusters.
    If `d_max = 1` this is equivalent to taking the edges of the graph.
    Then adds next-nearest neighbors and so on.
    """
    clusters = []
    distances = np.asarray(graph.distances())
    size = distances.shape[0]
    for i in range(size):
        for j in range(i + 1, size):
            if distances[i][j] <= d_max:
                clusters.append((i, j))

    res_clusters = np.empty((len(clusters), 2), dtype=np.int64)

    for i, cluster in enumerate(clusters):
        res_clusters[i] = np.asarray(cluster)

    return res_clusters






def tJExchangeRule(
    *,
    clusters: Optional[list[list[int]]] = None,
    graph: Optional[AbstractGraph] = None,
    d_max: int = 1,
):
    r"""
        Adapted from netket.sampler.rules.exchange
    """
    if clusters is None and graph is not None:
        clusters = compute_clusters(graph, d_max)
    elif not (clusters is not None and graph is None):
        raise ValueError(
            """You must either provide the list of exchange-clusters or a netket graph, from
                          which clusters will be computed using the maximum distance d_max. """
        )

    return tJExchangeRule_(jnp.array(clusters))


@struct.dataclass
class tJExchangeRule_(MetropolisRule):
    r"""
        Adapted from netket.sampler.rules.exchange
    """

    clusters: Any
        
    def random_state(self, sampler, machine, params, state, rng):
      apply_machine = apply_chunked(machine.apply, in_axes=(None, 0), chunk_size=4096)

      def loop_body(val):
          n, x, neg_inf_mask, params, rng, sampler = val
          n += 1
          probs = apply_machine(params, x).real
          neg_inf_mask = jnp.isneginf(probs) | jnp.isnan(probs) #| (jnp.abs(probs) >= 35) 
          num_trues = jnp.sum(neg_inf_mask)
          jax.debug.print("Number of inf values: {num}", num=(num_trues, jnp.sum(jnp.isnan(probs))))

          neg_inf_mask = jnp.repeat(jnp.expand_dims(neg_inf_mask, 1), x.shape[1], 1)
          rng = rng + sampler.n_batches*2
          #x_rep = sampler.hilbert.random_state(rng, size=sampler.n_batches, dtype=sampler.dtype)
          x_rep = self.random_state_(sampler, rng)
          x = jax.lax.select(neg_inf_mask, x_rep, x)
          return n, x, neg_inf_mask, params, rng, sampler

      def loop_cond(val):
          n, _, neg_inf_mask, _, _, _ = val
          neg_inf_mask = neg_inf_mask[:,0]
          return jnp.logical_and(jnp.any(neg_inf_mask), jnp.logical_or(jnp.sum(neg_inf_mask) > neg_inf_mask.shape[0]-1, n < 100))

      x = self.random_state_(sampler, rng)
      
      probs = apply_machine(params, x).real
      jax.debug.print("inital probs={z}",z=(probs, jnp.max(probs), jnp.min(probs)))
      neg_inf_mask = jnp.isneginf(probs) | jnp.isnan(probs) #| (jnp.abs(probs) >= 35) 
      neg_inf_mask = jnp.repeat(jnp.expand_dims(neg_inf_mask, 1), x.shape[1], 1)
      initial_val = (0, x, neg_inf_mask, params, rng, sampler)

      # Execute the while loop
      final_val = jax.lax.while_loop(loop_cond, loop_body, initial_val)
      _, x, neg_inf_mask, _, _, _ = final_val
      jax.debug.print('sum={x}', x=neg_inf_mask.shape)
      jax.debug.print('batches={x}', x=sampler.n_batches)

      # replace all -infs by first non -inf sample (loop through more rows if not converged)
      probs = apply_machine(params, x).real
      neg_inf_mask = jnp.isneginf(probs) | jnp.isnan(probs) #| (jnp.abs(probs) >= 35)
      sorted_indices = jnp.argsort(probs)
      jax.debug.print('sorted={x}', x=(neg_inf_mask[sorted_indices[-1]],probs[sorted_indices[-1]]))
      x_rep = x[sorted_indices[-1]]
      neg_inf_mask = jnp.repeat(jnp.expand_dims(neg_inf_mask, 1), x.shape[1], 1)
      x = jnp.where(neg_inf_mask, x_rep, x)      
          
      probs = apply_machine(params, x).real
      neg_inf_mask = jnp.isneginf(probs) | jnp.isnan(probs) #| (jnp.abs(probs) >= 35) 
      neg_inf_mask = jnp.repeat(jnp.expand_dims(neg_inf_mask, 1), x.shape[1], 1)
      num_trues = jnp.sum(neg_inf_mask)
      jax.debug.print("Number of inf values: {num}", num=(num_trues, jnp.sum(jnp.isnan(probs))))
      jax.debug.print("unique samples: {sam}", sam=jnp.sum(jnp.unique(x, fill_value=0, size=sampler.n_batches, axis=0))/sampler.hilbert.n_fermions)
      jax.debug.print("nans={x}", x=jnp.sum(jnp.isnan(probs)))
      jax.debug.print("final probs={z}",z=(probs, jnp.max(probs), jnp.min(probs)))
      
      return x.astype(jnp.float64)

    def random_state_(self, sampler, rng):
        #random sample with single occupancy
        batch_size = sampler.n_batches
        keys = jax.random.split(rng, batch_size)
        indices = jax.vmap(lambda key: jax.random.permutation(key, jnp.arange(sampler.hilbert.n_orbitals)))(keys)
        n_up = sampler.hilbert.n_fermions_per_spin[0]
        n_dn = sampler.hilbert.n_fermions_per_spin[1]
        up_ind = indices[:,:n_up]
        dn_ind = indices[:, n_up:(n_up+n_dn)]+sampler.hilbert.n_orbitals
        
        states = jnp.zeros((batch_size, sampler.hilbert.size))
        rows = jnp.arange(states.shape[0])
        
        #insert up spins
        states = states.at[(rows, up_ind.T)].set(1)
        #insert down spins
        states = states.at[(rows, dn_ind.T)].set(1)
        return states
        
    def transition(rule, sampler, machine, parameters, state, key, σ, return_flipped=False):
        def flip_true(sigma):
            new_sigma = jnp.zeros(sigma.shape)
            new_sigma = new_sigma.at[:n_modes//2].set(sigma[n_modes//2:])
            new_sigma = new_sigma.at[n_modes//2:].set(sigma[:n_modes//2])
            return new_sigma

        def flip_false(sigma):
            return sigma

        def flip_single_true(sigma):
            ind = jnp.nonzero(sigma[:sigma.shape[0]//2],size=1)[0]
            new_sigma = sigma.copy()
            new_sigma = new_sigma.at[ind].set(0.0)
            new_sigma = new_sigma.at[ind+sigma.shape[0]//2].set(1.0)
            return new_sigma
        n_chains = σ.shape[0]
        n_modes = σ.shape[1]

        # compute a mask for the clusters that can be hopped
        hoppable_clusters = _compute_different_clusters_mask(rule.clusters, σ)

        keys = jnp.asarray(jax.random.split(key, n_chains))
        # we use shard_map to avoid the all-gather coming from the batched jnp.take / indexing

        N = σ.sum(axis=-1).mean().astype(jnp.int32)

        @partial(sharding_decorator, sharded_args_tree=(True, True, True))
        @jax.vmap
        def _update_samples(key, σ, hoppable_clusters):
            # pick a random cluster, taking into account the mask
            n_conn = hoppable_clusters.sum(axis=-1)
            cluster = jax.random.choice(
                key,
                a=jnp.arange(rule.clusters.shape[0]),
                p=hoppable_clusters,
                replace=True,
            )

            flip = jax.random.choice(key, a=jnp.arange(2))
            # sites to be exchanged
            si = rule.clusters[cluster, 0]
            sj = rule.clusters[cluster, 1]

            σp = σ.at[si].set(σ[sj])
            σp = σp.at[sj].set(σ[si])
            σp = σp.at[si+n_modes//2].set(σ[sj + n_modes//2])
            σp = σp.at[sj+n_modes//2].set(σ[si + n_modes//2])
            # compute the number of connected sites
            hoppable_clusters_proposed = _compute_different_clusters_mask(
                rule.clusters, σp
            )

            # flip a single site if number of particles is odd
            σp_f = jax.lax.cond(jnp.logical_and(N%2==1,flip==0), flip_single_true, flip_false, σp)
            # flip the whole sample with a certain probability
            σp_flipped = jax.lax.cond(flip == 0, flip_true, flip_false, σp_f)

            # compute the number of connected sites
            n_conn_proposed = hoppable_clusters_proposed.sum(axis=-1)

            log_prob_corr = jnp.log(n_conn) - jnp.log(n_conn_proposed)
            #jax.debug.print("{x}",x=log_prob_corr)
            return σp, log_prob_corr, σp_flipped

        σps, log_prob_corrs, σps_flipped = _update_samples(keys, σ, hoppable_clusters)
        if return_flipped:
            return σps, log_prob_corrs, σps_flipped 
        else:
            return σps, log_prob_corrs #return _update_samples(keys, σ, hoppable_clusters)

    def __repr__(self):
        return f"ExchangeRule(# of clusters: {len(self.clusters)})"


@jax.jit
def _compute_different_clusters_mask(clusters, σ):
    # mask the clusters to include only moves
    # where the dof changes
    if jnp.issubdtype(σ, jnp.bool) or jnp.issubdtype(σ, jnp.integer):
        hoppable_clusters_mask = σ[..., clusters[:, 0]] != σ[..., clusters[:, 1]]
    else:
        N = σ.shape[-1]//2
        hoppable_clusters_mask = ~(jnp.isclose(σ[..., clusters[:, 0]], σ[..., clusters[:, 1]]) & jnp.isclose(σ[..., N + clusters[:, 0]], σ[..., N + clusters[:, 1]]))
        #hoppable_clusters_mask = ~jnp.isclose(
        #    σ[..., clusters[:, 0]], σ[..., clusters[:, 1]]
        #)
    return hoppable_clusters_mask
