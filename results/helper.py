import netket as nk
import jax
import numpy as np
from jax import numpy as jnp
from netket.experimental.operator.fermion import number as nc


def calculate_hole_corrs(L1, L2, g, hi, vstate, filename):
  def nh(site):
    up, down = +1, -1
    return 1-(nc(hi, site, up) + nc(hi, site, down))

  def append_hole_corrs(site1, site2, l, l_err, vstate):
    nhinhj = vstate.expect(nh(site1)*nh(site2))
    nhi = vstate.expect(nh(site1))
    nhj = vstate.expect(nh(site2)) 
    l.append(nhinhj.mean.real-nhi.mean*nhj.mean)
    err = np.sqrt(nhinhj.variance+nhi.mean**2*nhj.variance+nhj.mean**2*nhi.variance)
    l_err.append(err)
    return l, l_err


  nhnh_11, nhnh_11_err = [], []
  for i in g.nodes():
    x = i//L2
    y = i%L2
    xp = x+1
    yp = y+1
    if 0<= xp<L1 and 0<=yp<L2:
        j = xp*L2+yp
        nhnh_11, nhnh_11_err = append_hole_corrs(i, j, nhnh_11, nhnh_11_err, vstate)
  Cd_11 = [np.mean(nhnh_11),1/len(nhnh_11_err)*np.sqrt(np.sum(np.array(nhnh_11_err)**2))] 
  print(nhnh_11)
  print("Cd_11=", Cd_11)
  np.save(f'spin_correlations/nhnh11_'+filename.split("results/energy")[1]+'.npy', np.asarray(Cd_11))

  nhnh_10, nhnh_10_err = [], []
  for i in g.nodes():
    x = i//L2
    y = i%L2
    xp = x+1
    yp = y
    if 0<= xp<L1 and 0<=yp<L2:
        j = xp*L2+yp
        nhnh_10, nhnh_10_err = append_hole_corrs(i, j, nhnh_10, nhnh_10_err, vstate)
  Cd_10 = [np.mean(nhnh_10),1/len(nhnh_10_err)*np.sqrt(np.sum(np.array(nhnh_10_err)**2))] 
  np.save(f'spin_correlations/nhnh10_'+filename.split("results/energy")[1]+'.npy', np.asarray(Cd_10))
  print("Cd_10=", Cd_10)

  nhnh_20, nhnh_20_err = [], []
  for i in g.nodes():
    x = i//L2
    y = i%L2
    xp = x+2
    yp = y
    if 0<= xp<L1 and 0<=yp<L2:
        j = xp*L2+yp
        nhnh_20, nhnh_20_err = append_hole_corrs(i, j, nhnh_20, nhnh_20_err, vstate)
  Cd_20 = [np.mean(nhnh_20),1/len(nhnh_20_err)*np.sqrt(np.sum(np.array(nhnh_20_err)**2))] 
  print("Cd_20=", Cd_20)
  np.save(f'spin_correlations/nhnh20_'+filename.split("results/energy")[1]+'.npy', np.asarray(Cd_20))

  nhnh_21, nhnh_21_err = [], []
  for i in g.nodes():
    x = i//L2
    y = i%L2
    xp = x+2
    yp = y+1
    if 0<= xp<L1 and 0<=yp<L2:
        j = xp*L2+yp
        nhnh_21, nhnh_21_err = append_hole_corrs(i, j, nhnh_21, nhnh_21_err, vstate)
  Cd_21 = [np.mean(nhnh_21),1/len(nhnh_21_err)*np.sqrt(np.sum(np.array(nhnh_21_err)**2))] 
  print("Cd_21=", Cd_21)
  np.save(f'spin_correlations/nhnh21_'+filename.split("results/energy")[1]+'.npy', np.asarray(Cd_21))

  nhnh_22, nhnh_22_err = [], []
  for i in g.nodes():
    x = i//L2
    y = i%L2
    xp = x+2
    yp = y+2
    if 0<= xp<L1 and 0<=yp<L2:
        j = xp*L2+yp
        nhnh_22, nhnh_22_err = append_hole_corrs(i, j, nhnh_22, nhnh_22_err, vstate)
  Cd_22 = [np.mean(nhnh_22),1/len(nhnh_22_err)*np.sqrt(np.sum(np.array(nhnh_22_err)**2))] 
  print("Cd_22=", Cd_22)
  np.save(f'spin_correlations/nhnh22_'+filename.split("results/energy")[1]+'.npy', np.asarray(Cd_22))

  return Cd_11, Cd_10, Cd_20, Cd_21, Cd_22


def calculate_spin_corrs(L1, L2, g, hi, vstate, filename):
  def Sz(site):
    up, down = +1, -1
    return 1/2*(nc(hi, site, up) - nc(hi, site, down))

  def append_spin_corrs(site1, site2, l, l_err, vstate):
    SziSzj = vstate.expect(Sz(site1)*Sz(site2))
    Sz_i = vstate.expect(Sz(site1))
    Sz_j = vstate.expect(Sz(site2))
    val = ((SziSzj.mean-Sz_i.mean*Sz_j.mean)/(np.sqrt(Sz_i.variance)*np.sqrt(Sz_j.variance)))
    l.append(val.real)
    err = SziSzj.variance+Sz_i.mean**2*Sz_j.variance+Sz_i.variance*Sz_j.mean**2
    err = np.sqrt(err)/(np.sqrt(Sz_i.variance)*np.sqrt(Sz_j.variance))
    l_err.append(err)
    return l, l_err


  den = []
  for i in g.nodes():
    nup = vstate.expect(nc(hi, i, +1))
    ndn = vstate.expect(nc(hi, i, -1))
    den.append(np.array([nup.mean, ndn.mean, nup.error_of_mean, ndn.error_of_mean]))
  print("density=", den)
  np.save(f'spin_correlations/density_'+filename.split("results/energy")[1]+'.npy', np.asarray(den))

  szsz_11, szsz_11_err = [], []
  for i in g.nodes():
    x = i//L2
    y = i%L2
    xp = x+1
    yp = y+1
    if 0<= xp<L1 and 0<=yp<L2:
        j = xp*L2+yp
        szsz_11, szsz_11_err = append_spin_corrs(i, j, szsz_11, szsz_11_err, vstate)
  Cd_11 = [np.mean(szsz_11),1/len(szsz_11_err)*np.sqrt(np.sum(np.array(szsz_11_err)**2))] 
  print(szsz_11)
  print("Cd_11=", Cd_11)
  np.save(f'spin_correlations/szsz11_'+filename.split("results/energy")[1]+'.npy', np.asarray(Cd_11))

  szsz_10, szsz_10_err = [], []
  for i in g.nodes():
    x = i//L2
    y = i%L2
    xp = x+1
    yp = y
    if 0<= xp<L1 and 0<=yp<L2:
        j = xp*L2+yp
        szsz_10, szsz_10_err = append_spin_corrs(i, j, szsz_10, szsz_10_err, vstate)
  Cd_10 = [np.mean(szsz_10),1/len(szsz_10_err)*np.sqrt(np.sum(np.array(szsz_10_err)**2))] 
  np.save(f'spin_correlations/szsz10_'+filename.split("results/energy")[1]+'.npy', np.asarray(Cd_10))
  print("Cd_10=", Cd_10)

  szsz_20, szsz_20_err = [], []
  for i in g.nodes():
    x = i//L2
    y = i%L2
    xp = x+2
    yp = y
    if 0<= xp<L1 and 0<=yp<L2:
        j = xp*L2+yp
        szsz_20, szsz_20_err = append_spin_corrs(i, j, szsz_20, szsz_20_err, vstate)
  Cd_20 = [np.mean(szsz_20),1/len(szsz_20_err)*np.sqrt(np.sum(np.array(szsz_20_err)**2))] 
  print("Cd_20=", Cd_20)
  np.save(f'spin_correlations/szsz20_'+filename.split("results/energy")[1]+'.npy', np.asarray(Cd_20))

  szsz_21, szsz_21_err = [], []
  for i in g.nodes():
    x = i//L2
    y = i%L2
    xp = x+2
    yp = y+1
    if 0<= xp<L1 and 0<=yp<L2:
        j = xp*L2+yp
        szsz_21, szsz_21_err = append_spin_corrs(i, j, szsz_21, szsz_21_err, vstate)
  Cd_21 = [np.mean(szsz_21),1/len(szsz_21_err)*np.sqrt(np.sum(np.array(szsz_21_err)**2))] 
  print("Cd_21=", Cd_21)
  np.save(f'spin_correlations/szsz21_'+filename.split("results/energy")[1]+'.npy', np.asarray(Cd_21))

  szsz_22, szsz_22_err = [], []
  for i in g.nodes():
    x = i//L2
    y = i%L2
    xp = x+2
    yp = y+2
    if 0<= xp<L1 and 0<=yp<L2:
        j = xp*L2+yp
        szsz_22, szsz_22_err = append_spin_corrs(i, j, szsz_22, szsz_22_err, vstate)
  Cd_22 = [np.mean(szsz_22),1/len(szsz_22_err)*np.sqrt(np.sum(np.array(szsz_22_err)**2))] 
  print("Cd_22=", Cd_22)
  np.save(f'spin_correlations/szsz22_'+filename.split("results/energy")[1]+'.npy', np.asarray(Cd_22))

  return Cd_11, Cd_10, Cd_20, Cd_21, Cd_22


def calculate_polaron_corrs(L1, L2, g, hi, vstate, filename):
  def Sz(site):
    up, down = +1, -1
    return 1/2*(nc(hi, site, up) - nc(hi, site, down))

  # generate samples
  samples = vstate.sample(n_samples=4096*8*8)
  samples = samples.reshape(-1, 2*L1*L2)
  print(samples.shape)

  o_sites, Si, Sj = [],[],[]
  for i in g.nodes():
    x = i//L2
    y = i%L2
    x1 = x-1
    y1 = y
    x2 = x-1
    y2 = y-1
    if 0<=x1<L1 and 0<=y2<L2:
        j = x1*L2+y1
        l = x2*L2+y2
        op = Sz(j)*Sz(l)
        #eta = 1/np.sqrt(vstate.expect(Sz(j)).variance*vstate.expect(Sz(l)).variance)
        # select all samples with hole at site i
        mask = (np.abs(samples[:, i]) + np.abs(samples[:, hi.size // 2 + i])) == 0
        samples_ps = samples[mask]
        print("number of samples postselected:", samples_ps.shape)
        if samples_ps.shape[0]>0:
          #get matrix elements and connected states for all samples
          _, mel = op.get_conn_padded(samples_ps) #output: mel (n_samples, n_conn), x_conn (n_samples, n_conn, n_orbitals)
          oloc = np.sum(mel, axis=1) #sum over sigma prime
          o_sites.append((oloc).reshape((1,-1))) 
          _, si = Sz(j).get_conn_padded(samples_ps)
          si = np.sum(si, axis=1)
          Si.append((si).reshape((1,-1)))
          _, sj = Sz(l).get_conn_padded(samples_ps)
          sj = np.sum(sj, axis=1)
          Sj.append((sj).reshape((1,-1)))
  o_sites = jnp.concatenate(o_sites, axis=1).reshape((-1,))
  Si = jnp.concatenate(Si, axis=1).reshape((-1,))
  Sj = jnp.concatenate(Sj, axis=1).reshape((-1,))
  eta = 1/np.sqrt(np.var(Si)*np.var(Sj))
  C_h_10 = [np.mean(o_sites)*eta, eta*1/len(o_sites)*np.sqrt(np.var(o_sites))] 
  print(C_h_10)

  np.save(f'spin_correlations/polaron_corr_10'+filename.split("results/energy")[1]+'.npy', np.asarray(C_h_10))

  o_sites, Si, Sj = [],[],[]
  for i in g.nodes():
    x = i//L2
    y = i%L2
    x1 = x-1
    y1 = y
    x2 = x
    y2 = y-1
    count = 0
    if 0<=x1<L1 and 0<=y2<L2:
        j = x1*L2+y1
        l = x2*L2+y2
        op = Sz(j)*Sz(l)
        #eta = 1/np.sqrt(vstate.expect(Sz(j)).variance*vstate.expect(Sz(l)).variance)
        mask = (np.abs(samples[:, i]) + np.abs(samples[:, hi.size // 2 + i])) == 0
        samples_ps = samples[mask]
        print("number of samples postselected:",samples_ps.shape)
        if samples_ps.shape[0]>0:
          """x_conn, mel = op.get_conn_padded(samples_ps) #output: mel (n_samples, n_conn), x_conn (n_samples, n_conn, n_orbitals)
          # reshape x_conn to pass to log_value function
          x_conn_rs = x_conn.reshape(-1, x_conn.shape[-1]) #(n_samples*n_conn, n_orbitals)
    
          # get log probabilities
          log_samples = vstate.log_value(samples_ps)
          log_conn_rs = vstate.log_value(x_conn_rs)
        
          #reshape log_sigma_prime back
          log_conn = log_conn_rs.reshape(x_conn.shape[0], x_conn.shape[1])
          log_ratios = np.exp(log_conn-log_samples[:,None])
          oloc = np.sum(mel*log_ratios, axis=1)
          o_sites.append((oloc).reshape((1,-1)))"""
          #get matrix elements and connected states for all samples
          _, mel = op.get_conn_padded(samples_ps) #output: mel (n_samples, n_conn), x_conn (n_samples, n_conn, n_orbitals)
          oloc = np.sum(mel, axis=1) #sum over sigma prime
          o_sites.append((oloc).reshape((1,-1))) 
          _, si = Sz(j).get_conn_padded(samples_ps)
          si = np.sum(si, axis=1)
          Si.append((si).reshape((1,-1)))
          _, sj = Sz(l).get_conn_padded(samples_ps)
          sj = np.sum(sj, axis=1)
          Sj.append((sj).reshape((1,-1)))
  o_sites = jnp.concatenate(o_sites, axis=1).reshape((-1,))
  Si = jnp.concatenate(Si, axis=1).reshape((-1,))
  Sj = jnp.concatenate(Sj, axis=1).reshape((-1,))
  eta = 1/np.sqrt(np.var(Si)*np.var(Sj)) 
  print(eta, np.mean(o_sites), o_sites)
  C_h_11 = [np.mean(o_sites)*eta, eta*1/len(o_sites)*np.sqrt(np.var(o_sites))] 
  print(C_h_11)

  np.save(f'spin_correlations/polaron_corr_11'+filename.split("results/energy")[1]+'.npy', np.asarray(C_h_11))

  N = int(filename.split("Nup=")[1].split("_")[0]) + int(filename.split("Ndn=")[1].split("_")[0])
  nh = 1-(N/(L1*L2))
  o_sites, o_err = [], []
  for i in g.nodes():
    x = i//L2
    y = i%L2
    x1 = x-1
    y1 = y
    x2 = x
    y2 = y-1
    x3 = x+1
    y3 = y
    x4 = x
    y4 = y+1
    count = 0
    if 0<=x1<L1 and 0<=y2<L2 and  0<=x3<L1 and 0<=y4<L2:
        j = x1*L2+y1
        l = x2*L2+y2
        m = x3*L2+y3
        n = x4*L2+y4
        op = Sz(j)*Sz(l)*Sz(m)*Sz(n)*(1-(nc(hi, i, +1)+nc(hi, i, -1)))
        oloc = vstate.expect(op)
        hole_den = vstate.expect(1-(nc(hi, i, +1)+nc(hi, i, -1)))
        nh = hole_den.mean
        nh_var = hole_den.variance
        print("nh",nh,hole_den.mean, "oloc", oloc.mean)
        if nh!=0:
          o_sites.append(2**4/nh*oloc.mean)
          o_err.append(2**4*np.sqrt(oloc.variance/(nh**2) + oloc.mean**2/(nh_var**2))) #MC sum over samples
  C_h_1111 = [np.mean(o_sites), 1/len(o_err)*np.sqrt(np.sum(np.array(o_err)**2))] 
  print("5 point",C_h_1111)

  np.save(f'spin_correlations/polaron_corr_1111'+filename.split("results/energy")[1]+'.npy', np.asarray(C_h_1111))
