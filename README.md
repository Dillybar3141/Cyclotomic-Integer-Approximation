# Cyclotomic Integer Approximation

This repository contains testing code used for the work

>
> William H. Pan, Dylan Roscow, and Netanel Raviv. Complex sensing: Reconstructing dense rational signals from few complex measurements. December 2025. Submitted to IEEE Signal Processing Letters.

In the project we encountered the following subproblem.
Take $n^\text{th}$ roots of unity $\zeta_n^k = (e^{(2\pi / n) i})^k$. 
Given a "cyclotomic integer" $z$, that is, a sum of $n^\text{th}$ roots of unity

$z = c_0\zeta_n^0 + c_1\zeta_n^1 + \cdots + c_{n - 1}\zeta_n^{n - 1}$

where $c_0, c_1, \ldots, c_{n - 1}$ are unknown integers, can we find the $c_0, c_1, \ldots, c_{n - 1}$? 
To our knowledge there exists no analytical method.
In this code we implement the LLL algorithm \[Lenstra, Lenstra, Lovász (1982)\] to recover $c_0, c_1, \ldots, c_{n - 1}$ and test its success rate against various random distirbutions over the cyclotomic integers.

For questions reach out to whpan \[at\] utexas \[dot\] edu or droscow \[at\] purdue \[dot\] edu.

## Instructions

The code requires the dependencies for [SageMath](https://doc.sagemath.org/html/en/installation/).

`lll.py` contains all helper functions used in testing. 
We plot testing data in `examples.ipynb`, which the user is free to experiment with.
The main function to be used is `run_trials` which takes as parameters
- `orders`: the list of `n`'s to test against.
- `gen` and `gen_param`: selection of the random generator and its parameter.
- `Ascale` and `Bscale`: hyperparameters for our LLL matrix. In the noiseless case, highest success rates are achieved when `Ascale` is to the highest allowed by `prec` and `Bscale` is set around 3. For the noisy case set $\text{to be added}$.
- `last_root`: auxiliary variable testing the necessity of including $\zeta_n^{n - 1}$.
- `num_trials`: number of trials.
- `noise`: 
- `seed`: set random generator seed.
- `prec`: number of bits to use to define numbers. Implemented with `gmpy2.mpfr`.

For more guidance on best values to use see the testing data in `TBD.csv`. 
