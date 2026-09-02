import argparse
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

import proj_grad
import proj_maxcut_newton as pmn
import proj_maxcut_qi_sun as pqs

_OPT_VALUES = None

def _load_opt_values():
    global _OPT_VALUES
    if _OPT_VALUES is None:
        root = Path(__file__).parent
        opt_files = [root / "rudy_opt.csv", root / "ising_opt.csv"]
        values = {}
        for path in opt_files:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "data_file" not in df.columns or "opt_val" not in df.columns:
                raise ValueError(f"Expected columns data_file,opt_val in {path}")
            values.update({str(k): float(v) for k, v in zip(df["data_file"], df["opt_val"])})
        _OPT_VALUES = values
    return _OPT_VALUES


def _relative_error(value, optimum):
    if optimum == 0 or not np.isfinite(optimum):
        return float("nan")
    return abs(value - optimum) / abs(optimum)

from statsmodels.stats.correlation_tools import corr_nearest

def load_weight_matrix(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not lines:
        raise ValueError(f"No data found in {path}")

    n, m = map(int, lines[0].split())
    A = np.zeros((n, n), dtype=float)

    for line in lines[1 : 1 + m]:
        u, v, w = line.split()
        i, j = int(u) - 1, int(v) - 1
        weight = float(w)
        A[i, j] += weight
        A[j, i] += weight

    return A


def prepare_input(A):
    A = 0.5 * (A + A.T)
    scale = max(1.0, np.max(np.abs(A)))
    A_scaled = A / scale
    np.fill_diagonal(A_scaled, 1.0)
    return A_scaled, scale


def obj(W, L, regularization_factor):
    eigenvalues, _ = np.linalg.eigh(W)
    return -0.25 * np.trace(W @ L) + regularization_factor * (np.trace(W) - eigenvalues[-1])


def grad_maxcut(W, L, regularization_factor):
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    v = eigenvectors[:, -1]
    return -0.25 * L + regularization_factor * (np.eye(W.shape[0]) - np.outer(v, v))


def run_single_experiment(
    data_file,
    *,
    regularization_factor=5.0,
    max_iter_newton=200,
    max_iter_pgd=200,
    tol=1e-4,
    projection_method="newton",
    solver=cp.SCS,
    verbose_cvxpy=False,
):
    A_unscaled = load_weight_matrix(data_file)
    n = A_unscaled.shape[0]
    L_unscaled = np.diag(A_unscaled @ np.ones(n)) - A_unscaled

    A, scale = prepare_input(A_unscaled)
    n = A.shape[0]
    L = np.diag(A @ np.ones(n)) - A

    W = cp.Variable((n, n), symmetric=True)
    constraints = [
        W >> 0,
        cp.diag(W) == np.ones(n),
    ]
    objective = cp.Maximize(0.25 * cp.trace(L @ W))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose_cvxpy)

    if W.value is None:
        raise RuntimeError(f"CVXPY did not return a solution for {data_file}")

    W_opt = np.array(W.value, dtype=float)

    print(f"Rank of semidefinite relaxation: {np.linalg.matrix_rank(W_opt)}")

    newton_iter_count = 0
    if projection_method == "newton":
        def proj(W_current):
            nonlocal newton_iter_count
            X_proj, inner_iters = pmn.proj_maxcut_newton(
                W_current,
                MAX_ITER_NEWTON=max_iter_newton,
                return_stats=True,
            )
            newton_iter_count += inner_iters
            return X_proj
    else:
        proj = lambda W_current: pqs.nearest_correlation_matrix(W_current)[0]
        #proj = corr_nearest

    f = lambda W_current: obj(W_current, L, regularization_factor)
    grad = lambda W_current: grad_maxcut(W_current, L, regularization_factor)

    start_wall = time.perf_counter()
    res = proj_grad.pgd_mon(
        x0=W_opt,
        f=f,
        grad=grad,
        proj=proj,
        max_iter=max_iter_pgd,
        TOL=tol,
        #barzilai_borwein=True,
    )
    wall_time = time.perf_counter() - start_wall

    optimum_values = _load_opt_values()
    optimum = optimum_values.get(str(data_file), float("nan"))
    cvxpy_value = float(0.25 * np.trace(W_opt @ L_unscaled))
    pgd_value = float(0.25 * np.trace(res.x @ L_unscaled))
    cvxpy_rel = _relative_error(cvxpy_value, optimum)
    pgd_rel = _relative_error(pgd_value, optimum)
    return {
        "data_file": str(data_file),
        "n": n,
        "scale": scale,
        "optimal_value": float(optimum) if np.isfinite(optimum) else None,
        "cvxpy_status": prob.status,
        "cvxpy_value": cvxpy_value,
        "cvxpy_relative_error": cvxpy_rel,
        "pgd_status": res.status,
        "pgd_iterations": int(res.nit),
        "newton_iterations": int(newton_iter_count) if projection_method == "newton" else 0,
        "pgd_obj_value": pgd_value,
        "pgd_relative_error": pgd_rel,
        "pgd_outperforms_cvxpy": False if np.isnan(cvxpy_rel) or np.isnan(pgd_rel) else (np.abs(pgd_rel) < np.abs(cvxpy_rel)),
        "pgd_grad_norm": float(res.grad_norm),
        "cpu_time_seconds": float(res.cpu_time),
        "wall_time_seconds": float(wall_time),
        "message": res.message,
    }


def run_batch(
    data_dir,
    *,
    output_csv="maxcut_results.csv",
    pattern=None,
    **kwargs,
):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    files = sorted(data_dir.iterdir())
    if pattern is not None:
        files = [p for p in files if p.match(pattern)]
    files = [p for p in files if p.is_file()]

    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    rows = []
    for path in files:
        try:
            print(f"Processing {path}...")
            row = run_single_experiment(path, **kwargs)
            rows.append(row)
        except Exception as exc:
            rows.append({"data_file": str(path), "error": str(exc)})

    df = pd.DataFrame(rows)
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} rows to {output_csv}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Run the max-cut workflow on one file or a whole folder")
    parser.add_argument("--data-file", type=str, default="rudy_all/g05_60.0", help="Single input file to process")
    parser.add_argument("--data-dir", type=str, default=None, help="Folder containing many input files")
    parser.add_argument("--output", type=str, default="maxcut_results.csv", help="CSV file for batch results")
    parser.add_argument("--pattern", type=str, default=None, help="Optional glob pattern for batch files")
    parser.add_argument("--max-iter-pgd", type=int, default=200)
    parser.add_argument("--max-iter-newton", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--projection-method", type=str, default="newton", choices=["qi_sun", "newton"])
    parser.add_argument("--regularization-factor", type=float, default=3.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.data_dir is not None:
        df = run_batch(
            args.data_dir,
            output_csv=args.output,
            pattern=args.pattern,
            regularization_factor=args.regularization_factor,
            max_iter_newton=args.max_iter_newton,
            max_iter_pgd=args.max_iter_pgd,
            tol=args.tol,
            projection_method=args.projection_method,
        )
        print(df.head().to_string(index=False))
    else:
        result = run_single_experiment(
            args.data_file,
            regularization_factor=args.regularization_factor,
            max_iter_newton=args.max_iter_newton,
            max_iter_pgd=args.max_iter_pgd,
            tol=args.tol,
            projection_method=args.projection_method,
        )
        df = pd.DataFrame([result])
        df.to_csv(args.output, index=False)
        print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


