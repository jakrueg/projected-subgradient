from collections import deque
from dataclasses import dataclass


import time

import numpy as np


#TAU = 1e-5 # for RPCA
TAU = 0.1 # for mpec_style/MAXCUT

@dataclass()
class OptimizationResult:
    x: np.ndarray
    obj: float
    grad_norm: float
    nit: int
    status: int
    message: str
    cpu_time: float



class SlidingWindowMax:
    """max over the last `window_size` inserted values for non-monotone line search"""

    def __init__(self, window_size):
        self.window_size = window_size
        self.values = deque()

    def push(self, index, value):
        cutoff = index - self.window_size + 1
        while self.values and self.values[0][0] < cutoff:
            self.values.popleft()
        while self.values and self.values[-1][1] <= value:
            self.values.pop()
        self.values.append((index, value))

    def max(self):
        return self.values[0][1]

"""
Backtracking line search for projected gradient descent.
The parameter mu is the line search reference, which is either the function value in the monotone case,
or, in the nonmonotone case, the maximum of the last `memory` function values (for pgd_max) 
or a weighted average of the last function value and the previous mu (for pgd_avg).
"""
def pgd_map(proj, f, grad, x, mu,alpha_start=1.0):
    beta = 0.5
    c = 0.7

    MAX_BACKTRACKS = 40

    alpha = alpha_start
    grad_x = grad(x)
    y = proj(x-alpha*grad_x)
    for i in range(MAX_BACKTRACKS):
        f_y = f(y)
        if f_y <= mu + c * np.vdot(grad_x, y - x):
            return y, f_y
        else:
            alpha = alpha*beta
            argu = x - alpha * grad_x
            y = proj(argu)
    #print("Backtrack failed")
    return y, f(y)

"""
Projected gradient descent with max-rule nonmonotonicity
"""
def pgd_max(x0, f, grad, proj, memory=10, max_iter=100, TOL = 1e-6):
    if memory < 1:
        raise ValueError("memory must be at least 1")

    x = np.asarray(x0, dtype=float)
    f_x = f(x)
    mu = f_x


    f_window = SlidingWindowMax(memory)
    f_window.push(0, f_x)

    start_time = time.perf_counter()
    for k in range(max_iter):
        norm_iter_proj = np.max(np.abs(x  - proj(x - TAU*grad(x))))
        if norm_iter_proj < TOL:
            end_time =  time.perf_counter()
            cpu_time = end_time - start_time
            return OptimizationResult(x=x, obj=f_x, grad_norm=norm_iter_proj, nit=k, status=0, message="Converged", cpu_time=cpu_time )
        else:
            # Max-rule nonmonotonicity:
            # mu_k = max{f(x_j): max(0, k-memory+1) <= j <= k}
            mu = f_window.max()
            x, f_x = pgd_map(proj, f, grad, x, mu)
            f_window.push(k + 1, f_x)
    end_time =  time.perf_counter()
    cpu_time = end_time - start_time
    argu = x - TAU*grad(x)
    iter_proj = proj(argu)
    norm_iter_proj = np.max(np.abs(iter_proj-x))
    print(f"x-value: {x}")
    print(f"rank: {np.linalg.matrix_rank(x)}")
    print(f"Maximum absolute value: {np.max(np.abs(x))}, minimum absolute value: {np.min(np.abs(x))}")
    return OptimizationResult(x=x, obj=f_x, grad_norm=norm_iter_proj, nit=k, status=1, message="Maximum iterations reached", cpu_time=cpu_time )


"""
Projected gradient descent with average-rule nonmonotonicity
"""
def pgd_avg(x0, f, grad, proj, max_iter=100, p = 0.5,TOL = 1e-6,barzilai_borwein=False):
    alpha_min = 1e-20
    alpha_max = 1e20
    x = np.asarray(x0, dtype=float)
    f_x = f(x)
    mu = f_x
    start_time = time.perf_counter()
    for k in range(max_iter):
        g_x = grad(x)
        norm_iter_proj = np.max(np.abs(proj(x - TAU*g_x)-x))
        if norm_iter_proj < TOL:
            end_time =  time.perf_counter()
            cpu_time = end_time - start_time
            return OptimizationResult(x=x, obj=f_x, grad_norm=norm_iter_proj, nit=k, status=0, message="Converged", cpu_time=cpu_time )
                    
        else:
            if k==0 or not barzilai_borwein:
                alpha = 1.0
            else:
                s = x - x_old
                y = g_x - grad_old
                sy = np.vdot(s,y)
                yy = np.vdot(y,y)
                if yy == 0:
                    alpha = alpha_max
                else:
                    alpha = np.clip(sy/yy, alpha_min, alpha_max)
            x_old = x
            grad_old = g_x
            # average rule for nonmonotone line search:
            mu = (1-p)*mu + p*f_x
            x , f_x = pgd_map(proj, f, grad, x, mu,alpha_start=alpha)
    end_time =  time.perf_counter()
    cpu_time = end_time - start_time
    argu = x - TAU*g_x
    iter_proj = proj(argu)
    norm_iter_proj = np.max(np.abs(iter_proj-x))
    return OptimizationResult(x=x, obj=f_x, grad_norm=norm_iter_proj, nit=k, status=1, message="Maximum iterations reached", cpu_time=cpu_time )
"""
Projected gradient descent with monotone line search
"""        
def pgd_mon(x0, f, grad, proj, max_iter=100, TOL = 1e-6, barzilai_borwein=False):
    alpha_min = 1e-20
    alpha_max = 1e20
    x = np.asarray(x0, dtype=float)
    f_x = f(x)
    x_old = np.zeros_like(x)
    grad_old = np.zeros_like(x)
    start_time = time.perf_counter()
    for k in range(max_iter):
        g_x = grad(x)
        argu = x - TAU*g_x
        iter_proj = proj(argu)
        norm_iter_proj = np.max(np.abs(iter_proj-x))
        if norm_iter_proj < TOL:
            end_time =  time.perf_counter()
            cpu_time = end_time - start_time
            return OptimizationResult(x=x, obj=f_x, grad_norm=norm_iter_proj, nit=k, status=0, message="Converged", cpu_time=cpu_time )
        else:
            if k==0 or not barzilai_borwein:
                alpha = 1.0
            else:
                s = x - x_old
                y = g_x - grad_old
                sy = np.vdot(s,y)
                alpha = np.clip(sy/np.vdot(y,y), alpha_min, alpha_max)
            x_old = x
            grad_old = g_x
            # monotone gradient descent: mu = f(x)
            mu = f_x
            x , f_x = pgd_map(proj, f, grad, x, mu,alpha_start=alpha)
    end_time =  time.perf_counter()
    cpu_time = end_time - start_time
    argu = x - TAU*g_x
    iter_proj = proj(argu)
    norm_iter_proj = np.max(np.abs(iter_proj-x))
    return OptimizationResult(x=x, obj=f_x, grad_norm=norm_iter_proj, nit=k, status=1, message="Maximum iterations reached", cpu_time=cpu_time )

"""
Projected gradient descent with line search free adaptive step size selection by Yagishita and Ito (https://arxiv.org/abs/2509.14670)
"""
def pgd_ac(x0, f, grad, proj, max_iter=100, TOL = 1e-6):
    gamma = 1.0
    alpha = 1.3
    x = np.asarray(x0, dtype=float)
    start_time = time.perf_counter()
    for k in range(max_iter):
        grad_x = grad(x)
        argu = x - TAU*grad_x
        iter_proj = proj(argu)
        norm_iter_proj = np.max(np.abs(iter_proj-x))
        if norm_iter_proj < TOL:
            end_time =  time.perf_counter()
            cpu_time = end_time - start_time
            return OptimizationResult(x=x, obj=f(x), grad_norm=norm_iter_proj, nit=k, status=0, message="Converged", cpu_time=cpu_time )
                    
        else:
            tau = 1/(alpha*gamma)
            x_new = proj(x-tau*grad_x)
            kappa = 2*(f(x_new)-f(x)-np.vdot(grad_x,x_new-x))/np.vdot(x_new-x,x_new-x)
            gamma = max(gamma, kappa)
            x = x_new
    end_time =  time.perf_counter()
    cpu_time = end_time - start_time
    argu = x - TAU*grad_x
    iter_proj = proj(argu)
    norm_iter_proj = np.max(np.abs(iter_proj-x))
    return OptimizationResult(x=x, obj=f(x), grad_norm=norm_iter_proj, nit=k, status=1, message="Maximum iterations reached", cpu_time=cpu_time )
