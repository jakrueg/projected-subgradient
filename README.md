# projected-subgradient

Python implementation of several projected subgradient methods and numerical applications.

## Synopsis
This package contains implementations of projected subgradient methods for nonsmooth optimization problems of the following form.
```
   minimize        phi(x)
   over            x ∈ Rⁿ
   subject to      x ∈ D
```
where ``phi`` is an upper C^2 function and ``D`` is a non-empty, closed set where (possibly set-valued) projection operator is available.

The implementations in ``prox_grad.py`` differ in the way of stepsize selection:
* ```pgd_mon```: monotone linesearch
* ```pgd_avg```: nonmonotone linesearch using an average rule
* ```pgd_max```: nonmonotone linesearch using a max rule
* ```pgd_ac```: linesearch free auto-conditioned stepsize selection by Yagishita and Ito




## Bug reports and support

Please report any issues via the [Github issue tracker](https://github.com/jakrueg/SALM/issues). All types of issues are welcome including bug reports, typos, feature requests and so on.
