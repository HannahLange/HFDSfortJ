try:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  import jax
  jax.distributed.initialize()

  print(f"Rank={rank}: Total number of GPUs: {jax.device_count()}, devices: {jax.devices()}")
  print(f"Rank={rank}: Local number of GPUs: {jax.local_device_count()}, devices: {jax.local_devices()}", flush=True)

  # wait for all processes to show their devices
  comm.Barrier()
except:
  pass

import sys
sys.path.insert(1, '/project/th-scratch/h/Hannah.Lange/PhD/ML/HiddenFermions/src')
import argparse
import numpy as np
from jax import numpy as jnp
import netket as nk
import jax
from netket import experimental as nkx
import json
import optax
import os
import flax
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc


from hiddenfermions import *
from backflow import *
from exchange_new import *
from helper import *
from transformer import *


parser = argparse.ArgumentParser()
parser.add_argument("-Nx" , "--Nx"   , type=int,  default = 4 , help="length in x dir")
parser.add_argument("-Ny" , "--Ny"   , type=int,  default = 4 , help="length in y dir")
parser.add_argument("-Jz"  , "--Jz"    , type=float,default = 1. , help="spin-spin interaction")
parser.add_argument("-Jp"  , "--Jp"    , type=float,default = 1. , help="spin-spin interaction")
parser.add_argument("-t"  , "--t"    , type=float,default = 3. , help="hopping amplitude")
parser.add_argument("-Ne"  , "--n_elecs"    , type=int,default = 10 , help="number of electrons")
parser.add_argument("-b1"  , "--b1"    , type=int,default = 0 , help="boundary for x-dir (0:periodic, 1:open)")
parser.add_argument("-b2"  , "--b2"    , type=int,default = 0 , help="boundary for y-dir (0:periodic, 1:open)")
parser.add_argument("-init"  , "--MFinit"    , type=str, default = "Fermi" , help="initialization for MF")
parser.add_argument("-f"  , "--features"    , type=int, default = 32 , help="number of features for transformer / FFNN")
parser.add_argument("-l"  , "--layers"    , type=int, default = 1 , help="number of layers")
parser.add_argument("-nhid"  , "--nhid"    , type=int, default = 10 , help="number of hidden fermions")
parser.add_argument("-det"  , "--det"    , type=str, default = "hidden" , help="type of determinant: hidden or backflow")
parser.add_argument("-dtype"  , "--dtype"    , type=str, default = "real" , help="complex or real")

load = False

args = parser.parse_args()
L1      = args.Nx
L2      = args.Ny
n_elecs = args.n_elecs
Jz      = args.Jz
Jp      = args.Jp
t       = args.t
b1      = args.b1
b2      = args.b2
dtype   = args.dtype
determinant_type = args.det
bounds  = {1:{1:"OBC"}, 0:{0:"PBC"}}[b1][b2]
print("params: Jz=", Jz, "Jp=", Jp, "=", t, "Lx=", L1, "Lt=", L2, "bounds=", b1, b2)
MFinitialization = args.MFinit


# more parameters for the physical system
pbc     = [{0: True, 1:False}[b1],{0: True, 1:False}[b2]]
N_sites = L1*L2
N_up    = (n_elecs+1)//2
N_dn    = n_elecs//2

double_occupancy = False

# network parameters and sampling
lr               = 0.02
n_samples        = 4096*2
n_chains         = n_samples//2
n_steps          = 1000
n_hid            = args.nhid
features         = args.features
layers           = args.layers
n_modes          = 2*L1*L2
cs               = n_samples


# --------------- define the network -------------------
boundary_conditions = 'pbc' if pbc[0] else 'obc'
filename = f"results/energy_{L1}x{L2}_{boundary_conditions}x{boundary_conditions}_Nup={N_up}_Ndn={N_dn}_t={t}_Jz={Jz}_Jp={Jp}_lr={lr}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_nsamples={n_samples}_{determinant_type}_"+MFinitialization+"_"+dtype

g = nk.graph.Grid([L1,L2],pbc=pbc)
hi = nkx.hilbert.SpinOrbitalFermions(N_sites, s = 1/2, n_fermions_per_spin = (N_up, N_dn))
print(hi.size)

def Sz(site):
  return 1/2*(nc(hi, site, up) - nc(hi, site, down))

def Splus(site):
  return cdag(hi, site,up)*c(hi, site,down)

def Sminus(site):
  return cdag(hi, site,down)*c(hi, site,up)


if dtype=="real": dtype_ = jnp.float64
else: dtype_ = jnp.complex128
if determinant_type=="hidden":
    ma = HiddenFermion(n_elecs=n_elecs,
                   network="FFNN",
                   n_hid=n_hid,
                   Lx=L1,
                   Ly=L2,
                   layers=layers,
                   features=features,
                   double_occupancy_bool=double_occupancy,
                   MFinit=MFinitialization,
                   hilbert=hi,
                   stop_grad_mf=False,
                   stop_grad_lower_block=False,
                   bounds=bounds,
                   dtype=dtype_)
if determinant_type=="backflow":
    ma = Backflow(n_elecs=n_elecs,
                   network="FFNN",
                   Lx=L1,
                   Ly=L2,
                   layers=layers,
                   features=features,
                   double_occupancy_bool=double_occupancy,
                   MFinit=MFinitialization,
                   hilbert=hi,
                   stop_grad_mf=False,
                   bounds=bounds,
                   dtype=dtype_)

# ------------- define Hamiltonian ------------------------
up, down = +1, -1
ha = 0.0
for sz in (up, down):
    for u, v in g.edges():
        ha += -t*(cdag(hi, u, sz) * c(hi, v, sz)  + cdag(hi, v, sz) * c(hi, u, sz))

for u,v in g.edges():
    ha += Jz*Sz(u)*Sz(v)
    ha += 1/2*Jp*Splus(u)*Sminus(v)
    ha += 1/2*Jp*Sminus(u)*Splus(v)
    ha -= 1/4*Jz*(nc(hi,u,up) + nc(hi,u,down))*(nc(hi,v,up) + nc(hi,v,down))


# ---------- define sampler ------------------------
sa = nk.sampler.MetropolisSampler(hi, n_chains=n_chains, rule=tJExchangeRule(graph=g))

vstate = nk.vqs.MCState(sa, ma, n_samples=n_samples, chunk_size=cs, n_discard_per_chain=16) #defines the variational state object
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

# thermalize samples
for i in range(5):
    vstate.sample()

# -------------- start the training ---------------
if not load:
  schedule = optax.linear_schedule(init_value=1e-2, end_value=5e-4, transition_steps=n_steps-100)
  schedule_lr = optax.linear_schedule(init_value=lr, end_value=lr/2, transition_steps=n_steps-100)
  #sr = nk.optimizer.SR(diag_shift=schedule)
  op = nk.optimizer.Sgd(learning_rate=schedule_lr)
  #gs = nk.VMC(ha, op, variational_state=vstate, preconditioner=sr)
  gs = nkx.driver.VMC_SRt(ha, op, diag_shift=schedule, variational_state=vstate)
  gs.run(n_iter=n_steps-100, out=filename)

  with open("results/res"+filename.split("energy")[1]+".mpack", 'wb') as file:
    file.write(flax.serialization.to_bytes(vstate))


# stage 2
vstate = nk.vqs.MCState(sa, ma, n_samples=n_samples, chunk_size=cs, n_discard_per_chain=32) #defines the variational state object
with open(filename+".mpack", 'rb') as file:
  print("load vstate parameters")
  vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())
  #vstate = flax.serialization.from_bytes(vstate, file.read())
filename += "_2"
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

schedule = optax.linear_schedule(init_value=1e-2, end_value=5e-4, transition_steps=100)
schedule_lr = optax.linear_schedule(init_value=lr, end_value=lr/2, transition_steps=100)
op = nk.optimizer.Sgd(learning_rate=schedule_lr)
gs = nkx.driver.VMC_SRt(ha, op, diag_shift=schedule, variational_state=vstate)
gs.run(n_iter=100, out=filename)
with open("results/res"+filename.split("energy")[1]+".mpack", 'wb') as file:
    file.write(flax.serialization.to_bytes(vstate))


#calculate spin correlations
jax.lax.stop_gradient(calculate_spin_corrs(L1, L2, g, hi, vstate, filename))
jax.lax.stop_gradient(calculate_polaron_corrs(L1, L2, g, hi, vstate, filename))
