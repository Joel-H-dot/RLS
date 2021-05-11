# Overview

RLS is [available on PyPI][pypi], and can be installed via
```none
pip install RLS-OF
```
This package computes model parameters which minimise the error between measured and modelled data, where the sensitivity to model parameters is given (the Jacobian). This package computes the minimum of a regularised least squares objective function using a selection of trust region algorithms from the TRA package. Tikhonov regularisation is used, where the regularisation matrix can be a finite difference operator or the identity. Further, the NL2SOL algorithm can be chosen such that the Hessian is better approximated for large residual problems.  


[pypi]:  https://pypi.org/project/RLS-OF/

# Example Call
```
import RLS as regularised_least_squares
:
RLS_class = regularised_least_squares.RLS(measurement, initial_guess,
                                                       compute_Jacobian,
                                                       forward_compute, lower_constraint=0, upper_constraint=140e3, search_direction='Levenberg-Marquardt')
                                                      
minimum = self.RLS_class.compute_minimum()
```
Suppose we make a series of complex inductance measurements and we want to find the best fit with a model of a problem by tuning various material electrical conductivity variables. First, the real and imaginary parts of the data can be concatenated to give, effectively, twice the number of measurement points. Applying the RLS algorithm the fitted inductance curves:


![RLS_ex_fitted](https://user-images.githubusercontent.com/60707891/117804606-daec1e80-b24f-11eb-9032-c948cb863d92.png)

The predicted model paramters:

![RLS_ex](https://user-images.githubusercontent.com/60707891/117804657-ed665800-b24f-11eb-8234-a34f3e25628b.png)


# Theory 

For the theory behind the code see [[1]](#1), [[2]](#2) and [[3]](#3). 

## References
<a id="1">[1]</a> 
Hansen, P. C.  (1997). 
Rank-deficient and Discrete Ill-posed Problems: Numerical Aspects of Linear Inversion. 

<a id="2">[2]</a> 
Nocedal, J. and Wright, S.  (2006). 
Numerical Optimization. 

<a id="3">[3]</a> 
Conn, A. and Gould, N. and Toint, P.  (2000). 
Trust Region Methods.
