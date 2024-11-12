
This is the code for our work [Lange, Böhler et al., arXiv...](...). We use Hidden Fermion Determinant States as introduced by [Moreno et al. (2022)]([https://arxiv.org/abs/2406.00091](https://www.pnas.org/doi/10.1073/pnas.2122059119). 

The implementation is based on NetKet with the required packages listed in `requirements.txt`. The converged states for $6\times 10$ with open boundaries can be found in 'results'. They can be loaded by running

'python3 run.py -Nx 6 -Ny 10 -Jp 1.0 -Jz 1.0 -t 3.0 -tprime 0.0 -Ne 36 -b1 1 -b2 1 -f 128 -l 1 -nhid 20 -det hidden -dtype real -init Fermi'
