<div align="center">
    <img width="479" alt="Momentum_git" src="[https://github.com/HannahLange/HFDSfortJ/edit/main/HFDS.pdf]">
</div>


This is the code for our work [Lange, Böhler et al., arXiv...](...). We use Hidden Fermion Determinant States as introduced by [Moreno et al. (2022)](https://www.pnas.org/doi/10.1073/pnas.2122059119). 

Hidden Fermion Determinant States (HFDS) are based on a configuration dependent Slater determinant of physical and ancilla fermions, as sketched in the figure below. The implementation can be found in `src/hidden_fermions.py` or `src/hidden_fermions.py`. We recomment to use the latter which includes a global spin rotation symmetry. The implementation is based on NetKet with the required packages listed in `requirements.txt`. The converged states for $6\times 10$ with open boundaries can be found in `results/`. They can be loaded by running

`python3 run.py -Nx 6 -Ny 10 -Jp 1.0 -Jz 1.0 -t 3.0 -tprime 0.0 -Ne 36 -b1 1 -b2 1 -f 128 -l 1 -nhid 20 -det hidden -dtype real -init Fermi`
