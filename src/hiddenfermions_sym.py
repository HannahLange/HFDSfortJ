from jax import numpy as jnp
import netket as nk
import jax
from jax.random import PRNGKey, choice, split
from functools import partial
from flax import linen as nn
from jax.nn.initializers import zeros, normal, constant
from netket.utils.dispatch import dispatch
from netket import experimental as nkx
from netket.jax import apply_chunked
import numpy as np
from netket.hilbert.homogeneous import HomogeneousHilbert
from transformer import Transformer

class HiddenFermion(nn.Module):
  n_elecs: int
  network: str
  n_hid: int
  Lx: int
  Ly: int
  layers: int
  features: int
  double_occupancy_bool: bool
  MFinit: str
  hilbert: HomogeneousHilbert
  stop_grad_mf: bool = False
  stop_grad_lower_block: bool = False
  bounds: str="OBC"
  dtype: type = jnp.float64
  U: float=8.0

  def setup(self):
    self.n_modes = 2*self.Lx*self.Ly
    self.key = jax.random.PRNGKey(0)
    self.orbitals = Orbitals(self.n_elecs,self.n_hid,self.Lx, self.Ly, self.MFinit, self.stop_grad_mf, self.bounds, self.dtype, self.U)
    if self.network=="MorenoFFNN":
        self.hidden = [nn.Dense(features=self.features, use_bias=False, param_dtype=self.dtype) for i in range(self.n_hid)]
        self.output = [nn.Dense(features=(self.n_elecs + self.n_hid), use_bias=True, param_dtype=self.dtype) for i in range(self.n_hid)]
    elif self.network=="FFNN":
        self.hidden = [nn.Dense(features=self.features,use_bias=False,param_dtype=self.dtype) for i in range(self.layers)]
        self.output = nn.Dense(features=self.n_hid*(self.n_elecs + self.n_hid),use_bias=True,param_dtype=self.dtype)
    else:
        raise NotImplementedError()

    if self.MFinit!="random": 
        self.a = self.param('a', zeros, (1,), self.dtype)
        self.b = self.param('b', zeros, (3,), self.dtype) #needed if we couple two GPUs
  def double_occupancy(self,x):
    x = x[:,:x.shape[-1]//2] + x[:,x.shape[-1]//2:]

    return jnp.where(jnp.any(x > 1.5,axis=-1),True,False)


  def selu(self,x):
    if self.dtype==jnp.float64:
      return jax.nn.selu(x)
    else:
      return jax.nn.selu(x.real) +1j*jax.nn.selu(x.imag)


  def calc_psi(self,x,return_orbs=False):
    orbitals = self.orbitals(x)
    if self.network=="MorenoFFNN":
        outputs = [self.output[l](self.selu(self.hidden[l](x))) for l in range(len(self.hidden))]
        x_ = jnp.stack([o.reshape(x.shape[0],self.n_elecs + self.n_hid) for o in outputs], axis=1).reshape(x.shape[0],self.n_hid,self.n_elecs + self.n_hid)
    elif self.network=="FFNN":
        for i in range(self.layers):
            x = self.selu(self.hidden[i](x))
        x_ = self.output(x).reshape(x.shape[0],self.n_hid,self.n_elecs + self.n_hid)
    else:
        raise NotImplementedError()

    if self.MFinit!="random": x_ = self.a*x_
    if self.stop_grad_lower_block:
        x_ = jax.lax.stop_gradient(x_)
    x_ += jnp.concatenate((jnp.zeros((x.shape[0], self.n_hid, self.n_elecs),self.dtype), jnp.repeat(jnp.expand_dims(jnp.eye(self.n_hid), axis=0),x.shape[0],axis=0)), axis=2)
    x = jnp.concatenate((orbitals,x_),axis=1)
    sign, logx = jnp.linalg.slogdet(x)
    if return_orbs:
      return logx, jnp.log(sign + 0j), x
    else:
      return logx, jnp.log(sign + 0j)

  def flip_single_true(self, sigma):
    ind = jnp.nonzero(sigma[:sigma.shape[0]//2],size=1)[0]
    new_sigma = sigma.copy()
    new_sigma = new_sigma.at[ind].set(0.0)
    new_sigma = new_sigma.at[ind+sigma.shape[0]//2].set(1.0)
    return new_sigma
    
  def flip_single_false(self, sigma):
    return sigma

  def gen_reflected_samples(self,x):
    x = jax.lax.cond(self.n_elecs%2==1, self.flip_single_true, self.flip_single_false,x)
    x1 = x[:,:x.shape[1]//2].copy()
    x2 = x[:,(x.shape[1]//2):].copy()
    x_ = jnp.concatenate([x2,x1],axis=-1)
    return x_


  def __call__(self,x):
    do = self.double_occupancy(x)
    batch = x.shape[0]
    if self.n_elecs%2==0:
      x_refl    = self.gen_reflected_samples(x)
      log_psi, sign = self.calc_psi(jnp.concatenate([x,x_refl]))
      psi       = jnp.exp(log_psi)
      psi0      = psi[0:batch] 
      psi_refl  = psi[batch:] 
      log_psi = jnp.log(1/2*(psi0+psi_refl))+sign[0:batch]
    else:
      log_psi, sign = self.calc_psi(x)
      log_psi += sign
    if self.double_occupancy_bool:
      return log_psi
    else:
      return log_psi - 1e12*do

class Orbitals(nn.Module):
  n_elecs: int
  n_hid: int
  Lx: int
  Ly: int
  MFinit: str
  stop_grad_mf: bool
  bounds: str
  dtype: type = jnp.float64
  U: float=8.0

  def _init_orbitals_dct(self, key, shape, dtype):
    def ft_local_pbc(x,y,kx,ky):
      if self.dtype==jnp.float64:
        if kx<=self.Lx//2 and ky<=self.Ly//2:
            res = jnp.cos(2*jnp.pi*(x)/self.Lx*(kx))*jnp.cos(2*jnp.pi*(y)/self.Ly*(ky))
        elif kx>=self.Lx//2 and ky<=self.Ly//2:
            res = jnp.sin(2*jnp.pi*(x)/self.Lx*(kx))*jnp.cos(2*jnp.pi*(y)/self.Ly*(ky)) 
        elif kx<=self.Lx//2 and ky>=self.Ly//2:
            res = jnp.cos(2*jnp.pi*(x)/self.Lx*(kx))*jnp.sin(2*jnp.pi*(y)/self.Ly*(ky)) 
        elif kx>=self.Lx//2 and ky>=self.Ly//2:
            res = jnp.sin(2*jnp.pi*(x)/self.Lx*(kx))*jnp.sin(2*jnp.pi*(y)/self.Ly*(ky)) 
      else:
        res = jnp.exp(1j*2*jnp.pi*(kx/self.Lx*x + ky/self.Ly*y))
      return res


    def Hk(t):
      # define single particle Hamiltonian: literally have hopping from site to site
      N = self.Lx*self.Ly # number of sites
      H = jnp.zeros([N,N], dtype=self.dtype)
      for x in range(self.Lx):
        for y in range(self.Ly):
          i = x*self.Ly + y # map 1D to 2D
          # hopping
          if x<self.Lx-1:
            ix = (x+1)*self.Ly+y
            H = H.at[i,ix].add(-t)
            H = H.at[ix,i].add(-t)
          if y<self.Ly-1:
            iy = x*self.Ly + (y+1)
            H = H.at[i,iy].add(-t)
            H = H.at[iy,i].add(-t)
      en, us = jnp.linalg.eigh(H)
      return en, us


    def initialize_obc(num_particles,sigmaz):
      # 'orbitals' are now just eigenstates of single particle Hamiltonian (run from 0 to mX*mY-1) 
      ks = range(self.Lx*self.Ly)
      # find possible r-states
      rs = [[x,y] for y in range(self.Ly) for x in range(self.Lx)]

      mat = jnp.zeros([num_particles,len(rs)], dtype=self.dtype)
      # get single particle eigenenergies and states
      en, us = Hk(1)
      for i in range(num_particles):
        rcnt=0
        for r in rs:
          psi = us[rcnt,i] # wave function coefficient for this eigenstate + position r
          if self.dtype!=jnp.complex128: 
              assert np.isclose(jnp.imag(psi),0,1e-15)
              mat = mat.at[i, rcnt].set(np.real(psi)) 
          else:
              mat = mat.at[i, rcnt].set(psi) 
          rcnt+=1
      return mat


    def ft(k_arr, max_val,sigmaz):
      if "OBC" in self.bounds:
        try:
          matrix = initialize_obc(max_val,sigmaz)
        except AssertionError:
          jax.debug.print("Fall back to PBC")
          matrix = []
          for idx,(kx, ky) in enumerate(k_arr[:max_val]):
            kstate = [ft_local_pbc(x,y,kx,ky) for y in range(self.Ly) for x in range(self.Lx)]
            matrix.append(kstate)
      elif self.bounds=="PBC":
        matrix = []
        for idx,(kx, ky) in enumerate(k_arr[:max_val]):
          kstate = [ft_local_pbc(x,y,kx,ky) for y in range(self.Ly) for x in range(self.Lx)]
          matrix.append(kstate)
          #jax.debug.print("{x}",x=(-np.cos(2*np.pi*kx/self.Lx) - np.cos(2*np.pi*ky/self.Ly),kstate,kx,ky))
      return jnp.array(matrix)

    n_elecs = shape[1]
    k_modes = []
    for kx in range(0, self.Lx):
      for ky in range(0, self.Ly):
        k_modes.append((kx,ky))
    sorted_k_modes = sorted(k_modes, key=lambda x: (-np.cos(2*np.pi*x[0]/self.Lx) - np.cos(2*np.pi*x[1]/self.Ly), x))
    k_arr = np.array(sorted_k_modes)
    upmatrix = ft(k_arr, (n_elecs+1)//2,+1)
    dnmatrix = ft(k_arr, n_elecs//2,-1)
    mf = jnp.block([[upmatrix, jnp.zeros(upmatrix.shape)], [jnp.zeros(dnmatrix.shape),dnmatrix]]).T
    jax.debug.print("mf={x}",x=mf)
    #jax.debug.print("mf={x}",x=mf)
    return dtype(mf)


  def _init_orbitals_hartree(self, key, shape, dtype):
    mf = np.load("/project/th-scratch/h/Hannah.Lange/PhD/ML/HiddenFermions/src/orbs_8.0_16_14.npy")
    mf = jnp.array(mf)
    jax.debug.print("mf={x}",x=mf)
    return dtype(mf)


  @nn.compact
  def __call__(self,x):
    if self.MFinit=="Fermi":
        orbitals_mfmf = self.param('orbitals_mf',self._init_orbitals_dct,(2*self.Lx*self.Ly,self.n_elecs), self.dtype)
    elif self.MFinit=="Hartree":
        orbitals_mfmf = self.param('orbitals_mf',self._init_orbitals_hartree,(2*self.Lx*self.Ly,self.n_elecs), self.dtype)
    elif self.MFinit=="random":
        orbitals_mfmf = self.param('orbitals_mf', normal(0.1),(2*self.Lx*self.Ly,self.n_elecs), self.dtype)
    else:
        raise NotImplementedError("This MF initialization is not implemented! Chose one of: Fermi, random")
    orbitals_mfhf = self.param('orbitals_hf', zeros,(2*self.Lx*self.Ly,self.n_hid), self.dtype)
    if self.stop_grad_mf: 
        orbitals_mfmf = jax.lax.stop_gradient(orbitals_mfmf)
    orbitals = jnp.concatenate((orbitals_mfmf, orbitals_mfhf), axis=1)
    ind1, ind2 = jnp.nonzero(x,size=x.shape[0]*self.n_elecs)
    x = jnp.repeat(jnp.expand_dims(orbitals,0),x.shape[0],axis=0)[ind1,ind2]
    return x.reshape(-1,self.n_elecs,x.shape[1])
