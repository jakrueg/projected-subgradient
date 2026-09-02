from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
import math
import re

import time

import numpy as np
import sympy as sp


_INF = float("inf")

#TAU = 1e-5 # for PCA
TAU = 0.1 # for SMC/MAXCUT

@dataclass()
class OptimizationResult:
    x: np.ndarray
    obj: float
    grad_norm: float
    nit: int
    status: int
    message: str
    cpu_time: float


@dataclass(frozen=True)
class AmplVariable:
    name: str
    symbol_name: str
    lb: float
    ub: float
    init: float


@dataclass(frozen=True)
class PenaltyTerm:
    kind: str
    expr: sp.Expr
    name: str


@dataclass(frozen=True)
class ComplementarityPair:
    left: str
    right: str


class AmplProjectedProblem:
    """Projected-gradient input built from a small AMPL MPEC subset.

    Functional complementarity terms are rewritten as nonnegative slack
    variables with equality residuals. Plain constraints and slack-defining
    equalities are added to the objective with a quadratic penalty.
    """

    def __init__(
            self,
            variables,
            base_objective,
            penalty_terms,
            complementarity_pairs,
            penalty_weight=100.0,
            maximize=False):
        self.variables = list(variables)
        self.var_names = [v.name for v in self.variables]
        self.symbol_names = [v.symbol_name for v in self.variables]
        self.index = {name: i for i, name in enumerate(self.var_names)}
        self.base_objective_expr = -base_objective if maximize else base_objective
        self.penalty_terms = list(penalty_terms)
        self.complementarity_pairs = list(complementarity_pairs)
        self.penalty_weight = float(penalty_weight)

        symbols = [sp.Symbol(v.symbol_name) for v in self.variables]
        penalty_expr = sp.Integer(0)
        for term in self.penalty_terms:
            if term.kind == "eq":
                penalty_expr += term.expr ** 2
            elif term.kind == "le":
                penalty_expr += sp.Max(term.expr, 0) ** 2
            else:
                raise ValueError(f"unknown penalty term kind {term.kind!r}")

        self.penalized_objective_expr = (
            self.base_objective_expr + self.penalty_weight * penalty_expr
        )
        self._objective_fn = sp.lambdify(
            symbols, self.penalized_objective_expr, modules=["numpy", "math"]
        )
        self.gradient_expr = [
            sp.diff(self.penalized_objective_expr, symbol) for symbol in symbols
        ]
        self._gradient_fn = sp.lambdify(
            symbols, self.gradient_expr, modules=["numpy", "math"]
        )

        self._base_objective_fn = sp.lambdify(
            symbols, self.base_objective_expr, modules=["numpy", "math"]
        )

        self.x0 = np.array([v.init for v in self.variables], dtype=float)
        self.lb = np.array([v.lb for v in self.variables], dtype=float)
        self.ub = np.array([v.ub for v in self.variables], dtype=float)
        self.x0 = self.project(self.x0)

    def objective(self, x):
        x = np.asarray(x, dtype=float)
        return float(self._objective_fn(*x))

    def base_objective(self, x):
        x = np.asarray(x, dtype=float)
        return float(self._base_objective_fn(*x))

    def grad(self, x):
        """Symbolic gradient of the penalized objective."""
        x = np.asarray(x, dtype=float)
        return np.asarray(self._gradient_fn(*x), dtype=float)

    def finite_difference_grad(self, x, eps=1e-6):
        """Central finite-difference gradient, useful for checking grad()."""
        x = np.asarray(x, dtype=float)
        gradient = np.zeros_like(x)
        for i in range(x.size):
            step = eps * max(1.0, abs(x[i]))
            xp = x.copy()
            xm = x.copy()
            xp[i] += step
            xm[i] -= step
            gradient[i] = (self.objective(xp) - self.objective(xm)) / (2.0 * step)
        return gradient

    def project1(self, x):
        """Project onto variable bounds and simple complementarity pairs."""
        z = np.asarray(x, dtype=float)
        for pair in self.complementarity_pairs:
            i = self.index[pair.left]
            j = self.index[pair.right]
            flipped = False
            if self.ub[i] < self.ub[j]:
                flipped = True
                upper1 = self.ub[j]
                upper2 = self.ub[i]
                x1 = z[j]
                x2 = z[i]

            else:
                upper1 = self.ub[i]
                upper2 = self.ub[j]
                x1 = z[i]
                x2 = z[j]
            p1 = 0.0
            p2 = 0.0
            if x1<= 0 or x2 <= 0:
                p1 = np.maximum(x1,0.0)
                p2 = np.maximum(x2,0.0)
            elif x1 < upper2:
                if x1 <= x2:
                    p1 = 0.0
                    p2 = x2
                else:
                    p1 = x1
                    p2 = 0.0
            elif x1 < upper1:
                if x2 < (1/(2*upper2))*x1**2+0.5*upper2:
                    p1 = x1
                    p2 = 0.0
                else:
                    p1 = 0.0
                    p2 = upper2
            else:
                if x2 < (1/(2*upper2))*(upper2**2+2*x1*upper1 -upper1**2):
                    p1 = upper1
                    p2 = 0.0
                else:
                    p1 = 0.0
                    p2 = upper2
            if flipped:
                z[i] = np.clip(p2,0.0,upper2)
                z[j] = np.clip(p1,0.0,upper1)
            else:
                z[i] = np.clip(p1,0.0,upper1)
                z[j] = np.clip(p2,0.0,upper2)
        return np.clip(z, self.lb, self.ub)


    def project(self, x):
        """Project onto variable bounds and simple complementarity pairs."""
        #z = np.clip(np.asarray(x, dtype=float), self.lb, self.ub)
        z = np.asarray(x, dtype=float)
        for pair in self.complementarity_pairs:
            i = self.index[pair.left]
            j = self.index[pair.right]
            xi = z[i]
            xj = z[j]
            i_axis = z.copy()
            j_axis = z.copy()
            i_axis[i] = min(max(xi, self.lb[i]), self.ub[i])
            i_axis[j] = 0.0
            j_axis[i] = 0.0
            j_axis[j] = min(max(xj, self.lb[j]), self.ub[j])

            if ((i_axis[i] - xi) ** 2 + (i_axis[j] - xj) ** 2
                    <= (j_axis[i] - xi) ** 2 + (j_axis[j] - xj) ** 2):
                z[i], z[j] = i_axis[i], i_axis[j]
            else:
                z[i], z[j] = j_axis[i], j_axis[j]
        return np.clip(z, self.lb, self.ub)

    def values(self, x):
        """Return a dict mapping AMPL variable names to values."""
        return dict(zip(self.var_names, np.asarray(x, dtype=float)))


def load_ampl_problem(
        path,
        penalty_weight=100.0,
        objective_name=None,
        use_amplpy=True,
        data_paths=None):
    """Load a scalar AMPL MPEC model as projected-gradient callables.

    If amplpy is available, AMPL loads the model and expands variables, bounds,
    initial values, and simple sets. SymPy is then used for the penalized
    objective and its explicit gradient. If AMPL cannot be started, a smaller
    handwritten parser is used as a fallback for scalar models.
    """
    if use_amplpy:
        #try:
            return _AmplPySubsetParser(
                Path(path), penalty_weight, objective_name, data_paths
            ).parse()
        #except (ImportError, ModuleNotFoundError, RuntimeError, SystemError):
        #    pass
    #if data_paths:
    #    raise ValueError("data_paths require use_amplpy=True and a working AMPL runtime")
    #return _AmplSubsetParser(Path(path), penalty_weight, objective_name).parse()


class _AmplPySubsetParser:
    def __init__(self, path, penalty_weight, objective_name, data_paths):
        self.path = path
        self.penalty_weight = penalty_weight
        self.objective_name = objective_name
        if data_paths is None:
            self.data_paths = []
        elif isinstance(data_paths, (str, Path)):
            self.data_paths = [Path(data_paths)]
        else:
            self.data_paths = [Path(path) for path in data_paths]

    def parse(self):
        from amplpy import AMPL

        ampl = AMPL()
        try:
            ampl.option["presolve"] = 0
            ampl.read(str(self.path))
            for data_path in self.data_paths:
                ampl.read_data(str(data_path))

            parser = _AmplSubsetParser(self.path, self.penalty_weight, self.objective_name)
            parser.sets = self._sets(ampl)
            for variable in self._variables(ampl):
                parser._add_variable(
                    variable.name,
                    lb=variable.lb,
                    ub=variable.ub,
                    init=variable.init,
                )

            objective = None
            maximize = False
            for statement in self._expanded_statements(ampl):
                obj_match = _AmplSubsetParser._objective.match(statement)
                if obj_match:
                    sense, name, expr_text = obj_match.groups()
                    if ((self.objective_name is None and objective is None)
                            or name == self.objective_name):
                        objective = parser._expr(expr_text)
                        maximize = sense == "maximize"
                    continue
                if statement.startswith("subject to "):
                    parser._parse_constraint(statement)

            if objective is None:
                if self.objective_name is None:
                    raise ValueError(f"no objective found in {self.path}")
                raise ValueError(f"objective {self.objective_name!r} not found in {self.path}")

            return AmplProjectedProblem(
                parser.variables,
                objective,
                parser.penalty_terms,
                parser.complementarity_pairs,
                penalty_weight=self.penalty_weight,
                maximize=maximize,
            )
        finally:
            ampl.close()

    def _expanded_statements(self, ampl):
        output_handler = _AmplOutputCollector()
        previous_handler = getattr(ampl, "_output_handler", None)
        ampl.set_output_handler(output_handler)
        try:
            ampl.eval("expand;")
        finally:
            if previous_handler is not None:
                ampl.set_output_handler(previous_handler)
        return [
            _clean_ampl_statement(statement)
            for statement in _split_ampl_statements(output_handler.text)
        ]

    def _sets(self, ampl):
        sets = {}
        for name, ampl_set in ampl.get_sets():
            values = ampl_set.get_values().to_pandas().index.tolist()
            sets[name] = [_normalize_ampl_value(value) for value in values]
        return sets

    def _variables(self, ampl):
        variables = []
        for base_name, ampl_variable in ampl.get_variables():
            init_values = _entity_suffix_values(ampl_variable, "init", default=0.0)
            lb_values = _entity_suffix_values(ampl_variable, "lb0", default=-_INF)
            ub_values = _entity_suffix_values(ampl_variable, "ub0", default=_INF)
            for index, init in init_values.items():
                name = _ampl_instance_name(base_name, index)
                variables.append(AmplVariable(
                    name,
                    _sanitize_name(name),
                    _to_float(lb_values.get(index, -_INF)),
                    _to_float(ub_values.get(index, _INF)),
                    _to_float(init),
                ))
        return variables


class _AmplOutputCollector:
    def __init__(self):
        self.parts = []

    @property
    def text(self):
        return "".join(self.parts)

    def output(self, _, message):
        self.parts.append(message)


class _AmplSubsetParser:
    _var_decl = re.compile(r"^var\s+([A-Za-z_]\w*)(?:\s*\{([^}]*)\})?(.*)$", re.S)
    _set_decl = re.compile(r"^set\s+([A-Za-z_]\w*)\s*:=\s*([^;]+)$", re.S)
    _objective = re.compile(
        r"^(minimize|maximize)\s+([A-Za-z_]\w*)\s*:\s*(.+)$", re.S
    )
    _constraint = re.compile(
        r"^(?:subject\s+to\s+)?([A-Za-z_]\w*(?:\[[^\]]+\])?)?(?:\s*\{([^}]*)\})?\s*:\s*(.+)$",
        re.S,
    )

    def __init__(self, path, penalty_weight, objective_name):
        self.path = path
        self.penalty_weight = penalty_weight
        self.objective_name = objective_name
        self.sets = {}
        self.variables = []
        self.symbols = {}
        self.penalty_terms = []
        self.complementarity_pairs = []
        self._slack_count = 0

    def parse(self):
        statements = _split_ampl_statements(_strip_ampl_comments(self.path.read_text()))
        objective = None
        maximize = False

        for statement in statements:
            if not statement or statement == "subject to":
                continue
            set_match = self._set_decl.match(statement)
            if set_match:
                name, spec = set_match.groups()
                self.sets[name] = self._parse_index_set(spec.strip())
                continue

            var_match = self._var_decl.match(statement)
            if var_match:
                self._parse_var(var_match)
                continue

            obj_match = self._objective.match(statement)
            if obj_match:
                sense, name, expr_text = obj_match.groups()
                if ((self.objective_name is None and objective is None)
                        or name == self.objective_name):
                    objective = self._expr(expr_text)
                    maximize = sense == "maximize"
                continue

            if "complements" in statement or ":" in statement:
                self._parse_constraint(statement)
                continue

            raise ValueError(f"unsupported AMPL statement in {self.path}: {statement!r}")

        if objective is None:
            if self.objective_name is None:
                raise ValueError(f"no objective found in {self.path}")
            raise ValueError(f"objective {self.objective_name!r} not found in {self.path}")

        return AmplProjectedProblem(
            self.variables,
            objective,
            self.penalty_terms,
            self.complementarity_pairs,
            penalty_weight=self.penalty_weight,
            maximize=maximize,
        )

    def _parse_index_set(self, spec):
        range_match = re.match(r"^(-?\d+)\s*\.\.\s*(-?\d+)$", spec)
        if not range_match:
            raise ValueError(f"unsupported set declaration: {spec!r}")
        lo, hi = map(int, range_match.groups())
        return list(range(lo, hi + 1))

    def _parse_var(self, match):
        base_name, index_spec, tail = match.groups()
        if re.search(r"(?<![<>:])=(?!=)", tail):
            raise ValueError(f"defined variables are not supported: var {base_name}{tail}")
        indices = [None]
        if index_spec is not None:
            indices = self._indices(index_spec.strip())

        lb, ub, init = self._parse_var_tail(tail)
        for idx in indices:
            name = base_name if idx is None else f"{base_name}[{idx}]"
            self._add_variable(name, lb, ub, init)

    def _indices(self, spec):
        if spec in self.sets:
            return self.sets[spec]
        if " in " in spec or "," in spec:
            raise ValueError(f"unsupported indexed declaration {{{spec}}}")
        return self._parse_index_set(spec)

    def _parse_var_tail(self, tail):
        lb = -_INF
        ub = _INF
        init = 0.0

        lb_match = re.search(r">=\s*([^,]+?)(?=,|$)", tail)
        ub_match = re.search(r"<=\s*([^,]+?)(?=,|$)", tail)
        init_match = re.search(r":=\s*([^,]+?)(?=,|$)", tail)
        if lb_match:
            lb = self._number(lb_match.group(1).strip())
        if ub_match:
            ub = self._number(ub_match.group(1).strip())
        if init_match:
            init = self._number(init_match.group(1).strip())
        elif math.isfinite(lb) and lb > 0:
            init = lb
        elif math.isfinite(ub) and ub < 0:
            init = ub

        return lb, ub, init

    def _number(self, text):
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"only numeric bounds/initial values are supported, got {text!r}") from exc

    def _add_variable(self, name, lb=-_INF, ub=_INF, init=0.0):
        if name in self.symbols:
            raise ValueError(f"duplicate variable {name!r}")
        symbol_name = _sanitize_name(name)
        self.symbols[name] = sp.Symbol(symbol_name)
        self.variables.append(AmplVariable(name, symbol_name, lb, ub, init))

    def _parse_constraint(self, statement):
        match = self._constraint.match(statement)
        if not match:
            raise ValueError(f"unsupported constraint statement: {statement!r}")
        name, iterator, body = match.groups()
        name = name or f"constraint_{len(self.penalty_terms) + 1}"
        if iterator:
            loop_var, values = self._constraint_iterator(iterator)
            for value in values:
                indexed_body = _substitute_index(body, loop_var, value)
                self._parse_constraint(f"{name}_{value}: {indexed_body}")
            return

        if "complements" in body:
            left_text, right_text = re.split(r"\bcomplements\b", body, maxsplit=1)
            left_var = self._nonnegative_term_to_variable(name, "left", left_text)
            right_var = self._nonnegative_term_to_variable(name, "right", right_text)
            self.complementarity_pairs.append(ComplementarityPair(left_var, right_var))
            return

        for kind, residual in self._constraint_residuals(body):
            self.penalty_terms.append(PenaltyTerm(kind, residual, name))

    def _nonnegative_term_to_variable(self, constraint_name, side, text):
        expr = self._nonnegative_expr(text)
        direct_var = self._direct_nonnegative_variable(expr)
        if direct_var is not None:
            return direct_var

        slack_name = f"slack_{constraint_name}_{side}_{self._slack_count}"
        self._slack_count += 1
        self._add_variable(slack_name, lb=0.0, init=max(0.0, float(expr.evalf(subs={
            self.symbols[v.name]: v.init for v in self.variables if v.name in self.symbols
        })) if not expr.free_symbols else 0.0))
        slack_symbol = self.symbols[slack_name]
        self.penalty_terms.append(
            PenaltyTerm("eq", slack_symbol - expr, f"{constraint_name}_{side}_slack")
        )
        return slack_name

    def _nonnegative_expr(self, text):
        parts = _split_relational(text)
        if len(parts) == 1:
            return self._expr(parts[0])
        if len(parts) != 3:
            raise ValueError(f"unsupported complementarity side: {text!r}")
        left, op, right = parts
        if op == "<=":
            return self._expr(right) - self._expr(left)
        if op == ">=":
            return self._expr(left) - self._expr(right)
        raise ValueError(f"complementarity side must be an inequality: {text!r}")

    def _direct_nonnegative_variable(self, expr):
        for name, symbol in self.symbols.items():
            if expr == symbol:
                pos = self._var_pos(name)
                var = self.variables[pos]
                if var.lb < 0:
                    self.variables[pos] = replace(var, lb=0.0, init=max(0.0, var.init))
                return name
        return None

    def _var_pos(self, name):
        for i, var in enumerate(self.variables):
            if var.name == name:
                return i
        raise KeyError(name)

    def _constraint_residuals(self, text):
        parts = _split_relational(text)
        if len(parts) == 3:
            left, op, right = parts
            residual = self._expr(left) - self._expr(right)
            if op == "=":
                return [("eq", residual)]
            if op == "<=":
                return [("le", residual)]
            if op == ">=":
                return [("le", -residual)]
        if len(parts) == 5:
            lower, op1, middle, op2, upper = parts
            if op1 == "<=" and op2 == "<=":
                mid = self._expr(middle)
                return [("le", self._expr(lower) - mid), ("le", mid - self._expr(upper))]
            if op1 == ">=" and op2 == ">=":
                mid = self._expr(middle)
                return [("le", mid - self._expr(lower)), ("le", self._expr(upper) - mid)]
        raise ValueError(f"unsupported constraint body: {text!r}")

    def _expr(self, text):
        expr_text = _ampl_expr_to_sympy(text, self.symbols)
        try:
            return sp.sympify(expr_text, locals=self.symbols | _SYMPY_LOCALS)
        except Exception as exc:
            raise ValueError(f"could not parse expression {text!r} as {expr_text!r}") from exc

    def _constraint_iterator(self, iterator):
        match = re.match(r"^([A-Za-z_]\w*)\s+in\s+(.+)$", iterator.strip())
        if not match:
            raise ValueError(f"unsupported constraint iterator {{{iterator}}}")
        loop_var, spec = match.groups()
        return loop_var, self._indices(spec.strip())


_SYMPY_LOCALS = {
    "abs": sp.Abs,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "pi": sp.pi,
}


def _strip_ampl_comments(text):
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _clean_ampl_statement(text):
    text = _strip_ampl_comments(text).strip().rstrip(";").strip()
    return re.sub(r"\s+", " ", text)


def _split_ampl_statements(text):
    return [part.strip() for part in text.split(";") if part.strip()]


def _sanitize_name(name):
    return re.sub(r"\W", "__", name)


def _ampl_expr_to_sympy(text, symbols):
    rewritten = text.strip()
    for name in sorted(symbols, key=len, reverse=True):
        rewritten = rewritten.replace(name, symbols[name].name)
    rewritten = rewritten.replace("^", "**")
    return rewritten


def _split_relational(text):
    tokens = re.split(r"(<=|>=|==|=)", text.strip())
    return ["=" if token.strip() == "==" else token.strip()
            for token in tokens if token.strip()]


def _is_zero(text):
    try:
        return float(text.strip()) == 0.0
    except ValueError:
        return False


def _substitute_index(text, loop_var, value):
    rewritten = re.sub(rf"\[\s*{re.escape(loop_var)}\s*\]", f"[{value}]", text)
    return re.sub(rf"\b{re.escape(loop_var)}\b", str(value), rewritten)


def _normalize_ampl_value(value):
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return value
    if as_float.is_integer():
        return int(as_float)
    return as_float


def _to_float(value):
    if isinstance(value, str):
        if value.lower() == "inf":
            return _INF
        if value.lower() == "-inf":
            return -_INF
    return float(value)


def _entity_suffix_values(entity, suffix, default):
    try:
        data = entity.get_values(suffix).to_pandas()
    except RuntimeError:
        return _default_entity_values(entity, default)

    if data.shape[1] != 1:
        raise ValueError(f"unexpected AMPL suffix table for {entity.name()}.{suffix}")
    column = data.columns[0]
    return {index: data.loc[index, column] for index in data.index.tolist()}


def _default_entity_values(entity, default):
    if entity.is_scalar():
        return {0: default}
    try:
        indices = entity.get_values("init").to_pandas().index.tolist()
    except RuntimeError:
        indices = list(entity.instances())
    return {index: default for index in indices}


def _ampl_instance_name(base_name, index):
    if index == 0:
        return str(base_name)
    if isinstance(index, tuple):
        index_text = ",".join(str(_normalize_ampl_value(part)) for part in index)
    else:
        index_text = str(_normalize_ampl_value(index))
    return f"{base_name}[{index_text}]"


class SlidingWindowMax:
    """O(1) amortized max over the last `window_size` inserted values."""

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

def p2gd_map(proj, f, grad, x, mu):
    alpha_min = 1e-8
    alpha_max = 20
    beta = 0.5
    c = 0.7

    MAX_BACKTRACKS = 40

    alpha = 10
    grad_x = grad(x)
    g = proj(-grad_x)
    y = proj(x+alpha*g)
    for i in range(MAX_BACKTRACKS):
        f_y = f(y)
        if f_y <= mu - c * alpha * np.linalg.norm(g)**2:
            return y, f_y
        else:
            alpha = alpha*beta
            argu = x + alpha * g
            y = proj(argu)
    print("Backtrack failed")
    return y, f(y)


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
            #print(f"{k} iterations")
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

def pgd_avg(x0, f, grad, proj, max_iter=100, p = 0.5,TOL = 1e-6,barzilai_borwein=False):
    alpha_min = 1e-20
    alpha_max = 1e20
    x = np.asarray(x0, dtype=float)
    f_x = f(x)
    mu = f_x
    start_time = time.perf_counter()
    for k in range(max_iter):
        g_x = grad(x)
        #print(f"{k}-th PGD Iteration")
        norm_iter_proj = np.max(np.abs(proj(x - TAU*g_x)-x))
        if norm_iter_proj < TOL:
            #print(f"{k} iterations")
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
                #print(f"sy: {sy}")
                #if sy <= 0:
                #    alpha = alpha_max
                #else:
            #print(f"alpha: {alpha}")
            x_old = x
            grad_old = g_x
            mu = (1-p)*mu + p*f_x
            x , f_x = pgd_map(proj, f, grad, x, mu,alpha_start=alpha)
    end_time =  time.perf_counter()
    cpu_time = end_time - start_time
    argu = x - TAU*g_x
    iter_proj = proj(argu)
    norm_iter_proj = np.max(np.abs(iter_proj-x))
    #print(f"x-value: {x}")
    #print(f"rank: {np.linalg.matrix_rank(x)}")
    #print(f"Maximum absolute value: {np.max(np.abs(x))}, minimum absolute value: {np.min(np.abs(x))}")
    return OptimizationResult(x=x, obj=f_x, grad_norm=norm_iter_proj, nit=k, status=1, message="Maximum iterations reached", cpu_time=cpu_time )

def pgd_mon(x0, f, grad, proj, max_iter=100, TOL = 1e-6, barzilai_borwein=False):
    alpha_min = 1e-20
    alpha_max = 1e20
    x = np.asarray(x0, dtype=float)
    f_x = f(x)
    x_old = np.zeros_like(x)
    grad_old = np.zeros_like(x)
    start_time = time.perf_counter()
    for k in range(max_iter):
        #print(f"{k}-th PGD Iteration")
        g_x = grad(x)
        argu = x - TAU*g_x
        iter_proj = proj(argu)
        norm_iter_proj = np.max(np.abs(iter_proj-x))
        if norm_iter_proj < TOL:
        #if norm_g_x < TOL:
            #print(f"Converged after {k} iterations")
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
                #print(f"sy: {sy}")
                #if sy <= 0:
                #    alpha = alpha_max
                #else:
                alpha = np.clip(sy/np.vdot(y,y), alpha_min, alpha_max)
            #print(f"alpha: {alpha}")
            x_old = x
            grad_old = g_x
            mu = f_x
            x , f_x = pgd_map(proj, f, grad, x, mu,alpha_start=alpha)
            #print(f"{k}th iteration")
    end_time =  time.perf_counter()
    cpu_time = end_time - start_time
    argu = x - TAU*g_x
    iter_proj = proj(argu)
    norm_iter_proj = np.max(np.abs(iter_proj-x))
    print(f"x-value: {x}")
    print(f"rank: {np.linalg.matrix_rank(x)}")
    print(f"Maximum absolute value: {np.max(np.abs(x))}, minimum absolute value: {np.min(np.abs(x))}")
    return OptimizationResult(x=x, obj=f_x, grad_norm=norm_iter_proj, nit=k, status=1, message="Maximum iterations reached", cpu_time=cpu_time )

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
        #    print(f"{k} iterations")
            end_time =  time.perf_counter()
            cpu_time = end_time - start_time
            return OptimizationResult(x=x, obj=f(x), grad_norm=norm_iter_proj, nit=k, status=0, message="Converged", cpu_time=cpu_time )
                    
        else:
            tau = 1/(alpha*gamma)
            x_new = proj(x-tau*grad_x)
            kappa = 2*(f(x_new)-f(x)-np.vdot(grad_x,x_new-x))/np.vdot(x_new-x,x_new-x)
            gamma = max(gamma, kappa)
            x = x_new
            #print(f"{k}th iteration")
    end_time =  time.perf_counter()
    cpu_time = end_time - start_time
    argu = x - TAU*grad_x
    iter_proj = proj(argu)
    norm_iter_proj = np.max(np.abs(iter_proj-x))
    #print(f"x-value: {x}")
    #print(f"rank: {np.linalg.matrix_rank(x)}")
    #print(f"Maximum absolute value: {np.max(np.abs(x))}, minimum absolute value: {np.min(np.abs(x))}")
    return OptimizationResult(x=x, obj=f(x), grad_norm=norm_iter_proj, nit=k, status=1, message="Maximum iterations reached", cpu_time=cpu_time )
