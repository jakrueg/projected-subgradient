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

## Numerical Examples
* An MPEC style two-dimensional problem that is used to illustrate the importance of the particular stationarity conditions provided by the projected subgradient method.
* An implementation of the MAXCUT graph problem on data from the [Biq Mac library](https://biqmac.aau.at/biqmaclib.html).
* Robust principal component analysis for video background subtraction, applied to the [Change Detection 2014](https://changedetection.net/dataset2014/) dataset.


## Bug reports and support

Please report any issues via the [Github issue tracker](https://github.com/jakrueg/SALM/issues). All types of issues are welcome including bug reports, typos, feature requests and so on.
