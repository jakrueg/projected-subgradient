import numpy as np
from scipy.sparse.linalg import LinearOperator, minres

TOL = 1e-5

class _Converged(Exception):
    """Internal control-flow exception used to stop MINRES early
    once (4.2) or (4.3) is satisfied."""
    pass

def mat_vec_v(h, W, P):
    # Compute diag(P @ (W * (P.T @ diag(h) @ P)) @ P.T) more efficiently.
    Dmat = P.T @ (P * h[:, None])
    H = W * Dmat
    return np.sum((P @ H) * P, axis=1)


def f(A,y):
    W = A + np.diag(y)
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    eigenvalues = np.maximum(eigenvalues, 0)
    W_proj = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return 0.5*np.linalg.norm(W_proj, 'fro')**2-np.sum(y)

def grad_f(A,y):
    W = A + np.diag(y)
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    eigenvalues = np.maximum(eigenvalues, 0)
    W_proj = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return np.diag(W_proj)-1

def dual_to_primal(A,y):
    W = A + np.diag(y)
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    eigenvalues = np.maximum(eigenvalues, 0)
    W_proj = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return W_proj


def is_nearly_equal(a, b, gamma = 100):
    """Check if two numbers are nearly equal, with a tolerance that scales
    with the magnitude of the numbers."""
    return abs(a - b) <= gamma * (1 + abs(a) + abs(b)) * np.finfo(float).eps

def _cg_solve_cond(Vk_apply, grad_yk, eta_k, max_cg_iter, phi_k, sqrt_jacobi_cond):
    """
    Preconditioned conjugate gradient for the linear system

        V_k d = -grad_yk

    using a diagonal Jacobi-type preconditioner. The routine stops as soon
    as the residual-based test and the curvature-based test are both met.
    """
    grad_yk = np.asarray(grad_yk, dtype=float).ravel()
    n = grad_yk.shape[0]
    grad_norm = np.linalg.norm(grad_yk)
    if grad_norm <= 1e-30:
        return np.zeros(n)

    if sqrt_jacobi_cond is None:
        precond = np.ones(n)
    else:
        precond = np.asarray(sqrt_jacobi_cond, dtype=float)
        if precond.ndim == 2:
            precond = np.diag(precond)
        if precond.shape != (n,):
            precond = np.asarray(precond).reshape(-1)
            if precond.size != n:
                raise ValueError("preconditioner must have shape (n,) or (n, n)")

    def apply_precond(r):
        if precond is None:
            return r.copy()
        if precond.ndim == 2:
            return precond @ r
        return precond * r

    rhs = -grad_yk
    x = np.zeros(n)
    r = rhs - Vk_apply(x)
    z = apply_precond(r)
    p = z.copy()
    rz_old = float(np.dot(r, z))

    if rz_old <= 1e-30:
        return x

    for _ in range(max_cg_iter):
        Ap = Vk_apply(p)
        pAp = float(np.dot(p, Ap))
        if pAp <= 1e-14:
            break

        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        res_norm = np.linalg.norm(grad_yk + Vk_apply(x))
        d_norm = np.linalg.norm(x)

        cond_2_10 = res_norm <= min(eta_k, grad_norm) * grad_norm
        cond_2_11 = (
            d_norm > 1e-14 and
            (-np.dot(grad_yk, x) / (d_norm * d_norm)) >= min(phi_k, grad_norm)
        )
        if cond_2_10 and cond_2_11:
            return x

        z = apply_precond(r)
        rz_new = float(np.dot(r, z))
        if rz_new <= 1e-30:
            return x

        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    print("fallback")
    return x



def minres_cond(Vk_matvec, grad, Dk_diag, eta, varphi,
                maxiter=None, minres_tol=1e-10, rtol=None):
    """
    Compute the new direction d_k as in Step 5 using MINRES with a diagonal preconditioner. 
    
    """

def compute_direction(Vk_matvec, grad, Dk_diag, eta, varphi,
                       maxiter=None, minres_tol=1e-10, rtol=None):
    """
    Compute the new direction d_k as in Step 5.
 
    Parameters
    ----------
    Vk_matvec : callable
        Function v -> V_k @ v, computed via (2.5). `v` and the return
        value are 1-D numpy arrays of length n.
    grad : ndarray, shape (n,)
        The gradient grad f(y_k).
    Dk_diag : ndarray, shape (n,)
        The diagonal entries of the (diagonal) preconditioner D_k.
        Must be strictly positive.
    eta : float
        Tolerance parameter used in the residual-based test (4.2).
    varphi : float
        Tolerance parameter used in the curvature-based test (4.3).
    maxiter : int, optional
        Maximum number of MINRES iterations (passed straight to
        scipy). Defaults to scipy's own default (roughly 5*n).
    minres_tol : float
        Convergence tolerance passed to scipy's MINRES itself (this
        is independent of, and generally looser than, (4.2)/(4.3);
        it just bounds how long MINRES will run if our own tests
        never trigger).
    rtol : float, optional
        Newer versions of scipy use `rtol` instead of `tol`; if you
        hit a deprecation warning/error, set this instead of
        `minres_tol`.
 
    Returns
    -------
    d_k : ndarray, shape (n,)
        The computed direction.
    info : dict
        Diagnostic info: which condition triggered ('4.2', '4.3',
        'fallback'), number of MINRES iterations used, and the final
        residual norm achieved.
    """
    grad = np.asarray(grad, dtype=float).ravel()
    Dk_diag = np.asarray(Dk_diag, dtype=float).ravel()
    n = grad.size
 
    if np.any(Dk_diag <= 0):
        raise ValueError("Dk_diag must be strictly positive (it's a "
                          "diagonal preconditioner appearing under a "
                          "square root).")
 
    grad_norm = np.linalg.norm(grad)
 
    # Degenerate case: already at a stationary point.
    if grad_norm == 0.0:
        return np.zeros_like(grad), {"reason": "zero_gradient",
                                      "iterations": 0,
                                      "residual_norm": 0.0}
 
    # Operator corresponds to V_k, right-hand side is -grad.
    A = LinearOperator((n, n), matvec=Vk_matvec, dtype=float)
    rhs = -grad

    # Diagonal Jacobi preconditioner: provide M that approximately
    # applies the inverse of the diagonal preconditioner D_k.
    def M_matvec(v):
        return (1.0 / Dk_diag) * v
    M = LinearOperator((n, n), matvec=M_matvec, dtype=float)
 
    eta_term = min(eta, grad_norm)
    varphi_term = min(varphi, grad_norm)
 
    # State captured by the callback so we can report why we stopped.
    state = {"d": None, "reason": None, "iterations": 0,
              "residual_norm": None}
 
    def callback(u_k):
        # scipy's MINRES callback receives the current iterate u_k,
        # i.e. the approximate solution of V_k d = -grad, so u_k is d_k.
        state["iterations"] += 1
        d_k = u_k

        # --- test (4.2): ||grad f(y_k) + V_k d_k||_2 <= min(eta, ||grad||_2) * ||grad||_2
        residual = grad + Vk_matvec(d_k)
        residual_norm = np.linalg.norm(residual)
        state["residual_norm"] = residual_norm
        # Both criteria must hold to stop early.
        d_norm = np.linalg.norm(d_k)
        curvature = -(grad @ d_k) / (d_norm * d_norm) if d_norm > 0 else -np.inf
        if residual_norm <= eta_term * grad_norm and curvature >= varphi_term:
            state["d"] = d_k
            state["reason"] = "4.2+4.3"
            raise _Converged()
 
    kwargs = dict(callback=callback, maxiter=maxiter)
    tol_value = rtol if rtol is not None else minres_tol
    # scipy >= 1.12 renamed the `tol` kwarg to `rtol`; support both.
    import inspect
    tol_kw = "rtol" if "rtol" in inspect.signature(minres).parameters else "tol"
    kwargs[tol_kw] = tol_value
 
    try:
        u_sol, info_code = minres(A, rhs, M=M, **kwargs)
    except _Converged:
        return state["d"], {"reason": state["reason"],
                             "iterations": state["iterations"],
                             "residual_norm": state["residual_norm"]}
 
    # MINRES ran to (its own) convergence or maxiter without (4.2) or
    # (4.3) ever triggering in the callback -> check once more on the
    # final iterate, then fall back if still not satisfied.
    d_k = u_sol
    residual_norm = np.linalg.norm(grad + Vk_matvec(d_k))
    d_norm = np.linalg.norm(d_k)
    curvature = -(grad @ d_k) / (d_norm * d_norm) if d_norm > 0 else -np.inf
 
    if residual_norm <= eta_term * grad_norm and curvature >= varphi_term:
        return d_k, {"reason": "4.2+4.3", "iterations": state["iterations"],
                     "residual_norm": residual_norm}
 
    # Neither condition holds -> fallback as stated: d_k = -grad f(y_k)
    print(f"Fallback on Iteration")
    return -grad, {"reason": "fallback", "iterations": state["iterations"],
                    "residual_norm": residual_norm}
 


def proj_maxcut_newton(W0, MAX_ITER_NEWTON=200, return_stats=False):
    n = W0.shape[0]
    y = np.zeros(n)
    eta = 0.5
    phi = 1e-6
    mu = 0.5
    rho = 0.3
    sigma = 1e-4
    A = np.copy(W0)
    A = (A + A.T) / 2
    A[np.diag_indices_from(A)] = 1
    newton_iterations = 0
    for k in range(MAX_ITER_NEWTON):
        grad_f_y = grad_f(A, y)
        # termination check
        if np.linalg.norm(grad_f_y) < TOL:
            X_tilde = dual_to_primal(A, y)
            D = np.diag(np.diag(X_tilde))
            sqrt_D = np.sqrt(D)
            X = np.linalg.inv(sqrt_D) @ X_tilde @ np.linalg.inv(sqrt_D)
            return (X, newton_iterations) if return_stats else X
        newton_iterations += 1
    # eigenvalue decomposition of A + diag(y)
        eigenvalues, eigenvectors = np.linalg.eigh(A + np.diag(y))
        # Flip so that positive eigenvalues come first; keep eigenvectors in sync.
        eigenvalues = np.flip(eigenvalues)
        eigenvectors = np.flip(eigenvectors, axis=1)

        lam = eigenvalues
        lam_plus = np.maximum(lam, 0.0)
        tol_zero = 1e-12 * max(1.0, np.max(np.abs(lam)))
        alpha = np.where(lam > tol_zero)[0]
        beta = np.where(np.abs(lam) <= tol_zero)[0]
        gamma = np.where(lam < -tol_zero)[0]

        p = len(alpha)
        q = len(beta)
        r = len(gamma)

        pos_vals = lam[alpha]
        neg_vals = lam[gamma]
        T = pos_vals[:, None] / (pos_vals[:, None] - neg_vals[None, :]) if p and r else np.zeros((p, r))

        W = np.block([
            [np.ones((p, p)), np.ones((p, q)), T],
            [np.ones((q, p)), np.zeros((q, q)), np.zeros((q, r))],
            [T.T, np.zeros((r, q)), np.zeros((r, r))],
        ])

        P = eigenvectors
        Q = P**2
        v = np.sum((Q @ W) * Q, axis=1)

        if not np.any(lam_plus > 1e-12):
            # Zero operator case; MINRES will fall back if needed.
            Dk_diag = np.ones(n)
            def Vkd(h):
                return np.zeros(n)
        elif np.all(lam_plus > 1e-12):
            # Identity operator case.
            Dk_diag = np.ones(n)
            def Vkd(h):
                return h.copy()
        else:
            Dk_diag = 1.0 / v
            def Vkd(h):
                return mat_vec_v(h, W, P)

        d_k, info = compute_direction(
            Vkd,
            grad=grad_f_y,
            Dk_diag=Dk_diag,
            eta=eta,
            varphi=phi,
            maxiter=n,
        )
        # Diagnostic info from the inner solver (MINRES)
        #try:
        #    print(f"Newton iter {k}: compute_direction -> {info}")
        #except Exception:
        #    pass
        # info may contain diagnostic reason; if it fell back, d_k will be -grad_f_y
        # Armijo line search
        f_y = f(A, y)
        grad_y = grad_f_y
        grad_y_norm = np.linalg.norm(grad_y)
        alpha = 1.0
        for m in range(100):
            f_candidate = f(A, y + alpha * d_k)
            if f_candidate <= f_y + sigma * alpha * np.dot(grad_y, d_k):
                break
            if is_nearly_equal(f_candidate, f_y):
                grad_candidate = grad_f(A, y + alpha * d_k)
                if np.linalg.norm(grad_candidate) <= (1 - mu) * grad_y_norm:
                    alpha = 1.0
                    break
                else:
                    d_k = -grad_y
                    alpha = 1.0
            alpha *= rho
        else:
            print("Line search failed to find a suitable step size after 100 iterations.")
            break
        y = y+alpha*d_k
    X_tilde = dual_to_primal(A, y)
    # Normalize to unit diagonal safely (avoid division by zero).
    diag_vals = np.diag(X_tilde).astype(float)
    eps = 1e-12
    if np.any(diag_vals <= 0):
        # clamp small/non-positive diagonal entries to eps to avoid NaNs
        diag_safe = np.where(diag_vals > eps, diag_vals, eps)
    else:
        diag_safe = diag_vals
    inv_sqrt_diag = 1.0 / np.sqrt(diag_safe)
    X = (inv_sqrt_diag[:, None] * X_tilde) * inv_sqrt_diag[None, :]
    return (X, newton_iterations) if return_stats else X


"""
#if __name__ == "__main__":
    # quick self-test on a small random "almost correlation" matrix
np.random.seed(0)
n = 5
B = np.random.randn(n, n)
A = 0.5 * (B + B.T)
np.fill_diagonal(A, 1.0)

X = proj_maxcut_newton(A)

print("Resulting matrix X:\n", X)
#print("\nConverged:", info['converged'], "in", info['iterations'], "iterations")
print("Resulting diagonal (should be ~1):", np.diag(X))
eigvals = np.linalg.eigvalsh(X)
print("Eigenvalues of X (should be >= 0):", eigvals)
"""
