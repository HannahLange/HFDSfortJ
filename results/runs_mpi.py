import os
import numpy as np


size = [6,10]
bounds = [1,1]
Jp = 1.0
Jz = 1.0
t = 3.0

Lx, Ly = size
b1, b2 = bounds

det = "hidden"
dtype = "real"
if Lx*Ly==48: Nhs= [48,47,46,44,40,32,24,16,8,4]
elif Lx*Ly==36: Nhs =[36,35,34,32,28,24,20,16,12,6,4]
elif Lx*Ly==60: Nhs = [60,59,58,56,52,44,36,28,20,12,4]
for l in [1]:
  for features in [128]:
    for Ne in [59,58,56,52,44,36,28,20,12,4]:
      for init in ["random","Fermi"]:
        for nhid in [20]: 
            with open("mljob_mpi.sh", "r") as f:
                data = f.read()
            string = ".py -Nx {Lx} -Ny {Ly} -Jp {Jp} -Jz {Jz} -t {t} -Ne {Ne} -b1 {b1} -b2 {b2} -f {features} -l {l} -nhid {nhid} -det "+det+" -dtype "+dtype+" -init "
            string = string.format(Lx=Lx, Ly=Ly, Jp=Jp, Jz=Jz, t=t, Ne=Ne, b1=b1, b2=b2, features=features, l=l, nhid=nhid)
            data = data.replace(".py", string+init)
            with open("mljob_mpi_"+str(size[0])+"x"+str(size[1])+".sh", "w") as f:
                data = f.write(data)
            os.system("sbatch "+"mljob_mpi_"+str(size[0])+"x"+str(size[1])+".sh")
