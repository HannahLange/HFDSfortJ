# Simulating the t-J model with Hidden Fermion Determinant States
<div align="center">
    <img width="579" alt="HFDS" src="https://github.com/HannahLange/HFDSfortJ/blob/main/HFDS.png">
</div>


This is the code for our work [Lange, Böhler et al., arXiv:2411.10430](https://arxiv.org/abs/2411.10430). We use Hidden Fermion Determinant States as introduced by [Moreno et al. (2022)](https://www.pnas.org/doi/10.1073/pnas.2122059119). 

Hidden Fermion Determinant States (HFDS) are based on a configuration dependent Slater determinant of physical and ancilla fermions, as sketched in the figure above. The implementation can be found in `src/hidden_fermions.py` or `src/hidden_fermions.py`. We recommend to use the latter which includes a global spin rotation symmetry. The implementation is based on NetKet with the required packages listed in `requirements.txt`. The converged states for $6\times 10$ systems with open boundaries can be found in `results/` (without symmetry) and `results_sym/` (with symmetry). They can be loaded by running

`python3 run.py -Nx 6 -Ny 10 -Jp 1.0 -Jz 1.0 -t 3.0 -tprime 0.0 -Ne 36 -b1 1 -b2 1 -f 128 -l 1 -nhid 20 -det hidden -dtype real -init Fermi`
