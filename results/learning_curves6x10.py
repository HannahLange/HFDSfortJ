#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:18:34 2024

@author: hannah.lange
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:32:51 2022

@author: hannah.lange
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import os
import matplotlib.lines as mlines
import matplotlib as m
from scipy.optimize import curve_fit
import json

# Some parameters to make the plots look nice
params = {
    "text.usetex": True,
    "font.family": "serif",
    "legend.fontsize": 10,
    "figure.figsize": (5, 4.5),
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 3,
    "lines.markeredgewidth": 2,
    "lines.markersize": 0,
    "lines.marker": "o",
    "patch.edgecolor": "black",
}
plt.rcParams.update(params)
plt.style.use("seaborn-deep")


os.environ["PATH"] += os.pathsep + '/opt/local/bin'


cm = plt.get_cmap('tab20') 


def load_nqs_data(Lx, Ly, t, U, Nup, Ndn, nhid, bounds, nlayers, nfeatures,  ty, init, model="FH", nsamples=2048):
    energies, errors = [],[]
    for i in range(2):
        try:
            filename = "../6x10_psi_sym/results/energy_"+str(Lx)+"x"+str(Ly)+"_"+bounds+"x"+bounds+"_Nup="+str(Nup)+"_Ndn="+str(Ndn)+"_t="+str(t)+"_Jz="+str(U)+"_Jp="+str(U)+"_lr=0.02_nlayers="+str(nlayers)+"_nfeatures="+str(nfeatures)+"_nhid="+str(nhid)+"_nsamples="+str(nsamples)+"_"+str(ty)+"_"+init+"_real"
            if i>0: 
                filename+="_"+str(i+1)
 
            filename+=".log"
            data=json.load(open(filename))
            energy = np.array(data["Energy"]["Mean"]["real"])
            error = np.array(data["Energy"]["Variance"])
            energies.append(energy)
            errors.append(error)
        except:
            pass
    energies = np.concatenate(energies)
    errors = np.concatenate(errors)
    return energies[:1800].real, errors[:1800].real



def load_energies_DMRG(Lx, Ly, t, J, Nh):
    chis = [256,512,1024,2048,4096]
    dmrg = ['DMRG3S','DMRG3S','DMRG3S','DMRG3S','2DMRG']
    energies = []
    params = []
    for s,stage in enumerate(["gs_stage_1","gs_stage_2", "gs_stage_3", "gs_stage_4", "gs_final"]):
        if stage =="init":
            file_path = "../../DMRG_ED_comparison/DMRG/obs_init_state___H_exp__lat="+str(Lx)+"x"+str(Ly)+"_bc=OBC_Nh="+str(Nh)+"_S="+str(((Lx*Ly-Nh)%2)/2)+"_J="+str(J)+"_t="+str(t)+"_alpha=-0.25.txt" 
        else:
            file_path = "../../DMRG_ED_comparison/DMRG/obs_"+stage+"___H_exp__lat="+str(Lx)+"x"+str(Ly)+"_bc=OBC_Nh="+str(Nh)+"_S="+str(((Lx*Ly-Nh)%2)/2)+"_J="+str(J)+"_t="+str(t)+"_alpha=-0.25_m="+str(chis[s])+"_ver="+dmrg[s]+".txt" 
        with open(file_path, "r") as f:
            data = f.readlines()[0].split("\n")[0]
        data = complex(data).real
        energies.append(data)
        params.append(Lx*Ly*chis[s]**2*3)
    return params, np.array(energies)
        

t=3.0
J=1.0
Lx = 6
Ly = 10
nlayers = 1
bounds = "obc"
nfeatures = 128
nhid = 20

nsamples = 4096*2
bounds = "obc"
nfeatures = 128
nlayers = 1
Ns = [4,12, 16,20,28,36,  44, 52, 56, 58,59,60]
for ty in ["hidden"]:
    fig, ax = plt.subplots(2,len(Ns), figsize=(13,4), sharey="row", sharex=True)  
    
    for n,N in enumerate(Ns):
        c = 0
        print("-----------",N,"-----------")
        for nhid in [20,4,10, 24]:
            
            Nup = (N+1)//2
            Ndn = N//2
            
            
            for init in ["random",   "Fermi"]:
                try:
                    energy, var = load_nqs_data(Lx, Ly, t, J, Nup, Ndn, nhid, bounds,nfeatures=nfeatures, nlayers=nlayers,nsamples=nsamples,ty=ty, init=init)
                    _, E_ED = load_energies_DMRG(Lx, Ly, t, J, Lx*Ly-N)
                    E_ED=E_ED[-1]
                    ax[0,n].plot(np.arange(len(energy)),(energy-E_ED)/np.abs(E_ED), label="N="+str(N)+": "+ty+", "+init+", f="+str(nfeatures)+", $n_h$="+str(nhid)+", $N_s$="+str(nsamples), alpha=0.7,color=cm(c))
                    ax[1,n].plot(np.arange(len(energy)),np.sqrt(var)/np.abs(E_ED), label="N="+str(N)+": "+ty+", "+init+", f="+str(nfeatures)+", $n_h$="+str(nhid)+", $N_s$="+str(nsamples), alpha=0.7,color=cm(c)) 
                    print(ty,init,nhid, energy[-10:].mean(), E_ED, (energy[-10:].mean()-E_ED)/np.abs(E_ED), len(energy))
                    #ax.axhline(y=(energy[0]-e_ed)/np.abs(e_ed),xmin=0, xmax=(len(energy)), color=cm(c), linestyle="dashed")
                except:
                    pass
                
                c+= 1
            ax[0,n].set_title("N="+str(N))
            ax[0,n].grid(True)
            ax[1,n].grid(True)
        
            ax[0,0].set_ylabel("relative error") 
            #ax.legend()
    ax[0,0].semilogy([],[]) 
    ax[1,0].semilogy([],[]) 

    ax[0,0].set_ylim(1e-4, 2) 
    plt.xlim(0,1800)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    
    
    

    
    
