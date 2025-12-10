"""
УНИВЕРСАЛЬНЫЙ градиентный спуск с бэктрек-линейным поиском (Armijo) + CLI + реестр + ИНТЕРАКТИВ
+ ранняя остановка по стабилизации f + автовыбор стартовой точки + отчёт + 2D-визуализация (по запросу).

Фишки
-----
- Любая скалярная f(x: np.ndarray) -> float, R^n -> R.
- Градиент строится автоматически (JAX, если установлен; иначе центральная разность).
- Реестр готовых функций: rosenbrock, rosenbrock_nd, quadratic2d, rastrigin, himmelblau, beale, polyquartic2d.
- CLI + интерактив:
    * можно выбрать функцию из списка;
    * можно ввести свою формулу;
    * можно включить автоподбор начальной точки (--auto-x0 или вопрос в интерактиве).
- Ранняя остановка по стабилизации функции: если |Δf| мал на протяжении окна — стоп.

Алгоритм
--------
GD: x_{k+1} = x_k - α_k ∇f(x_k),  p_k = -∇f(x_k).
Armijo backtracking подбирает α_k: f(x+αp) ≤ f(x) + c α <∇f(x), p>.

Останов
-------
- ||∇f||_2 < tol_grad
- ||Δx||_2 < tol_step
- Стабилизация f: последние patience_f итераций имеют |Δf| < tol_f
- k >= max_iter

CLI примеры
-----------
python -m optimization.universal_gd                             # интерактив
python -m optimization.universal_gd --list                      # показать все функции
python -m optimization.universal_gd --func rosenbrock --x0 0,0
python -m optimization.universal_gd --func rosenbrock_nd --x0 2.4,1.5,2.1,0.7,1.1
python -m optimization.universal_gd --func polyquartic2d --auto-x0 --compare-scipy
python -m optimization.universal_gd --func rastrigin --x0 0,0,0,0,0 --grad-policy finite
"""

from dataclasses import dataclass
from typing import Callable, Tuple, List, Optional, Dict
import argparse
import sys
from math import isfinite
import numpy as np
from numpy.typing import NDArray
from collections import deque

# SciPy – опционально, только для сравнения
try:
    from scipy.optimize import minimize

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# =========================
#  Автогенерация градиента
# =========================

def _central_diff_grad(
        f: Callable[[NDArray[np.floating]], float],
        eps: float = 1e-8
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """N-мерная центральная разность (стабильно и без зависимостей)."""

    def g(x: NDArray[np.floating]) -> NDArray[np.floating]:
        x = np.asarray(x, dtype=float)
        n = x.size
        grad = np.empty(n, dtype=float)
        for i in range(n):
            h = eps * max(1.0, abs(x[i]))
            ei = np.zeros(n, dtype=float)
            ei[i] = 1.0
            try:
                fp = f(x + h * ei)
            except Exception:
                fp = np.inf
            try:
                fm = f(x - h * ei)
            except Exception:
                fm = np.inf
            grad[i] = (fp - fm) / (2.0 * h)
        return grad

    return g


def make_gradient(
        f: Callable[[NDArray[np.floating]], float],
        prefer: str = "auto",
        eps: float = 1e-8
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """
    Конструирует ∇f:
      - "auto": попробовать JAX, иначе central-diff
      - "jax":  JAX, при ошибке — fallback на central-diff
      - "finite": всегда central-diff
    """
    if prefer in ("auto", "jax"):
        try:
            import jax
            import jax.numpy as jnp

            def f_jax(x_jnp: "jnp.ndarray") -> "jnp.ndarray":
                return jnp.asarray(f(np.array(x_jnp, dtype=float)), dtype=jnp.float_)

            g_jax = jax.jit(jax.jacfwd(f_jax))

            def g(x: NDArray[np.floating]) -> NDArray[np.floating]:
                return np.array(g_jax(np.asarray(x, dtype=float)), dtype=float)

            return g
        except Exception:
            pass

    return _central_diff_grad(f, eps=eps)


# =========================
#  ЛИНЕЙНЫЙ ПОИСК (ARMIJO) — устойчивый
# =========================

def backtracking_armijo(
        f: Callable[[NDArray[np.floating]], float],
        g: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        x: NDArray[np.floating],
        p: NDArray[np.floating],
        alpha0: float = 1.0,
        c: float = 1e-4,
        rho: float = 0.5,
        max_trials: int = 20,
) -> tuple[float, int]:
    """
    Подбор шага α по Армихо:
        f(x + α p) ≤ f(x) + c α ⟨∇f(x), p⟩.
    Возвращает (alpha, reductions), где reductions — сколько раз уменьшали шаг.
    Устойчив к NaN/Inf/исключениям в f(x+αp) и переполнениям в скалярном произведении.
    """
    try:
        fx = f(x)
    except Exception:
        fx = np.inf
    gx = g(x)
    with np.errstate(over='ignore', invalid='ignore'):
        dot = float(np.dot(gx, p))
    if not np.isfinite(dot):
        dot = 0.0

    alpha = alpha0
    reductions = 0
    for _ in range(max_trials):
        try:
            with np.errstate(over='ignore', invalid='ignore'):
                f_trial = f(x + alpha * p)
        except Exception:
            f_trial = np.inf

        rhs = fx + c * alpha * dot
        if np.isfinite(f_trial) and f_trial <= rhs:
            return alpha, reductions

        alpha *= rho
        reductions += 1
    return alpha, reductions


# =========================
#  Вспомогательное: автовыбор x0
# =========================

def autoselect_x0(
        f: Callable[[NDArray[np.floating]], float],
        dim: int,
        samples: int = 64,
        radius: float = 5.0,
        seed: int = 42
) -> NDArray[np.floating]:
    """
    Выбирает стартовую точку как argmin по набору кандидатов:
      - 0-вектор, ±базисные, + случайные точки в [-radius, radius]^n.
    """
    rng = np.random.default_rng(seed)
    candidates: List[NDArray[np.floating]] = [np.zeros(dim, dtype=float)]
    for i in range(dim):
        e = np.zeros(dim);
        e[i] = 1.0
        candidates += [e.copy(), -e.copy()]
    for _ in range(max(0, samples - len(candidates))):
        candidates.append(rng.uniform(-radius, radius, size=dim))

    best_x = None
    best_f = np.inf
    for x in candidates:
        try:
            fx = float(f(x))
            if np.isfinite(fx) and fx < best_f:
                best_f, best_x = fx, x
        except Exception:
            continue
    return np.zeros(dim, dtype=float) if best_x is None else np.asarray(best_x, dtype=float)


# =========================
#  ГРАДИЕНТНЫЙ СПУСК
# =========================

@dataclass(slots=True)
class GradientDescent:
    f: Callable[[NDArray[np.floating]], float]
    g: Optional[Callable[[NDArray[np.floating]], NDArray[np.floating]]] = None
    step: float = 1.0
    tol_grad: float = 1e-8
    tol_step: float = 1e-12
    tol_f: float = 1e-12  # стабилизация функции
    patience_f: int = 500
    max_iter: int = 100_000
    use_backtracking: bool = True
    grad_policy: str = "auto"
    fd_eps: float = 1e-8
    log_every: int = 200

    # служебные поля последней итерации (для отчета)
    _last_alpha: float = 0.0
    _last_bt_reductions: int = 0
    _last_grad_norm: float = 0.0
    _stop_reason: str = "max_iter"

    def _grad(self) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
        return self.g if self.g is not None else make_gradient(self.f, prefer=self.grad_policy, eps=self.fd_eps)

    def run(self, x0: NDArray[np.floating] | Tuple[float, ...]) -> Tuple[
        NDArray[np.floating], float, int, List[float], dict, NDArray[np.floating]]:
        """
        Возвращает: (x_star, f_star, iters, history_f, meta, trajectory)
        meta = { 'stop_reason', 'grad_norm', 'alpha', 'bt_reductions' }
        trajectory: массив shape (m, n) — точки траектории по итерациям (для визуализации).
        """
        x = np.asarray(x0, dtype=float)
        gfun = self._grad()
        history_f: List[float] = []
        df_window: deque[float] = deque(maxlen=max(1, self.patience_f))
        traj: List[NDArray[np.floating]] = [x.copy()]

        self._stop_reason = "max_iter"
        self._last_alpha = 0.0
        self._last_bt_reductions = 0
        self._last_grad_norm = 0.0

        for k in range(1, self.max_iter + 1):
            gk = gfun(x)
            ng = float(np.linalg.norm(gk))
            fk = self.f(x)
            history_f.append(fk)

            if not isfinite(fk) or not np.isfinite(gk).all():
                raise FloatingPointError("f/grad not finite; diverged.")

            # критерий 1: по норме градиента
            if ng < self.tol_grad:
                self._stop_reason = "grad_tol"
                self._last_grad_norm = ng
                break

            # критерий 2: стабилизация f
            if len(history_f) >= 2:
                df = abs(history_f[-1] - history_f[-2])
                df_window.append(df)
                if len(df_window) == df_window.maxlen and all(d <= self.tol_f for d in df_window):
                    self._stop_reason = "f_stabilized"
                    self._last_grad_norm = ng
                    break

            p = -gk
            if self.use_backtracking:
                alpha, red = backtracking_armijo(self.f, gfun, x, p, alpha0=self.step)
            else:
                alpha, red = self.step, 0

            x_new = x + alpha * p

            # критерий 3: по длине шага
            if np.linalg.norm(x_new - x) < self.tol_step:
                x = x_new
                traj.append(x.copy())
                self._stop_reason = "step_tol"
                self._last_alpha = alpha
                self._last_bt_reductions = red
                self._last_grad_norm = ng
                break

            # записываем «последнее известное»
            self._last_alpha = alpha
            self._last_bt_reductions = red
            self._last_grad_norm = ng

            x = x_new
            traj.append(x.copy())

            if self.log_every and (k % self.log_every == 0):
                print(f"[{k}] f={fk:.6e}, ||g||={ng:.3e}, alpha={alpha:.2e}, bt_reductions={red}, x={x}")

        meta = {
            "stop_reason": self._stop_reason,
            "grad_norm": self._last_grad_norm,
            "alpha": self._last_alpha,
            "bt_reductions": self._last_bt_reductions,
        }
        return x, float(self.f(x)), k, history_f, meta, np.asarray(traj)


# =========================
#  РЕЕСТР ФУНКЦИЙ
# =========================

@dataclass(frozen=True)
class FuncSpec:
    name: str
    func: Callable[[NDArray[np.floating]], float]
    dim: Optional[int]  # None => любой n
    description: str
    minima_note: Optional[str] = None  # подсказка про минимум(ы)


def rosenbrock_nd(x: NDArray[np.floating]) -> float:
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def rosenbrock_2d(x: NDArray[np.floating]) -> float:
    x1, x2 = float(x[0]), float(x[1])
    return (1.0 - x1) ** 2 + 100.0 * (x2 - x1 ** 2) ** 2


def quadratic2d(x: NDArray[np.floating]) -> float:
    x1, x2 = float(x[0]), float(x[1])
    return (x1 - 2.0) ** 2 + (x2 + 1.0) ** 2


def rastrigin(x: NDArray[np.floating]) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    return 10.0 * n + np.sum(x ** 2 - 10.0 * np.cos(2 * np.pi * x))


def himmelblau(x: NDArray[np.floating]) -> float:
    a = x[0] ** 2 + x[1] - 11.0
    b = x[0] + x[1] ** 2 - 7.0
    return float(a * a + b * b)


def beale(x: NDArray[np.floating]) -> float:
    x1, x2 = float(x[0]), float(x[1])
    return (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (2.625 - x1 + x1 * x2 ** 3) ** 2


def polyquartic2d(x: NDArray[np.floating]) -> float:
    x1, x2 = float(x[0]), float(x[1])
    return 2.0 * x1 ** 2 - 4.0 * x1 * x2 + x2 ** 4 + 2.0


FUNC_REGISTRY: Dict[str, FuncSpec] = {
    "rosenbrock": FuncSpec("rosenbrock", rosenbrock_2d, 2, "Rosenbrock 2D: (1-x)^2 + 100(y-x^2)^2",
                           "Глобальный минимум в (1, 1)."),
    "rosenbrock_nd": FuncSpec("rosenbrock_nd", rosenbrock_nd, None,
                              "Rosenbrock nD: sum 100(x_{i+1}-x_i^2)^2 + (1-x_i)^2", "Глобальный минимум в (1,…,1)."),
    "quadratic2d": FuncSpec("quadratic2d", quadratic2d, 2, "Quadratic 2D: (x-2)^2 + (y+1)^2",
                            "Глобальный минимум в (2, -1)."),
    "rastrigin": FuncSpec("rastrigin", rastrigin, None, "Rastrigin nD: 10n + sum(x^2 - 10 cos 2πx)",
                          "Много локальных минимумов, глобальный в (0,…,0)."),
    "himmelblau": FuncSpec("himmelblau", himmelblau, 2, "Himmelblau 2D",
                           "Четыре глобальных минимума: (3,2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)."),
    "beale": FuncSpec("beale", beale, 2, "Beale 2D", "Глобальный минимум в (3, 0.5)."),
    "polyquartic2d": FuncSpec("polyquartic2d", polyquartic2d, 2, "2x^2 - 4xy + y^4 + 2",
                              "Два глобальных минимума: (1,1) и (-1,-1)."),
}


def list_functions() -> str:
    lines = []
    for k, spec in FUNC_REGISTRY.items():
        dim = "any" if spec.dim is None else str(spec.dim)
        lines.append(f"- {k:13s} | dim={dim} | {spec.description}")
    return "\n".join(lines)


# =========================
#  SAFE eval для пользовательской функции (Py3.13 friendly)
# =========================

_ALLOWED_GLOBALS = {
    "np": np,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs, "pi": np.pi, "e": np.e,
}


def build_custom_function(expr: str, dim: int) -> Callable[[NDArray[np.floating]], float]:
    """
    Создает f(x)->float из текстового выражения.
    Поддерживает стиль с x[i] и x0,x1,… (y≡x1, z≡x2).
    """
    expr = expr.strip()

    # Заглушка для __import__: защищает от любых попыток импорта внутри eval
    def _no_import(*args, **kwargs):
        raise ImportError("Import is disabled in this sandboxed eval")

    def f(x: NDArray[np.floating]) -> float:
        x = np.asarray(x, dtype=float)
        if x.size != dim:
            raise ValueError(f"custom f expects dim={dim}, got {x.size}")

        locals_map = {"x": x}
        for i in range(dim):
            locals_map[f"x{i}"] = x[i]
        if dim >= 2:
            locals_map["y"] = x[1]
        if dim >= 3:
            locals_map["z"] = x[2]

        # Py3.13-safe: подсовываем __import__ как вызываемую заглушку
        safe_globals = {"__builtins__": {"__import__": _no_import}, **_ALLOWED_GLOBALS}

        val = eval(expr, safe_globals, locals_map)  # noqa: S307 (осознанно, sandbox-глобалы)
        return float(val)

    return f


# =========================
#  Визуализация 2D (по запросу)
# =========================

def visualize_2d_contours_and_path(
        f: Callable[[NDArray[np.floating]], float],
        traj: NDArray[np.floating],
        minima_hint: Optional[str] = None,
        title: str = "Gradient Descent Trajectory (2D)"
) -> None:
    """
    Рисует 2D-контуры f и поверх — траекторию GD. Требует matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib не установлен — визуализация пропущена.")
        return

    if traj.shape[1] != 2:
        print("[WARN] Визуализация доступна только для 2D. Пропускаю.")
        return

    # разумные границы: по траектории + небольшой отступ
    xmin, ymin = traj.min(axis=0) - 1.0
    xmax, ymax = traj.max(axis=0) + 1.0
    # если траектория маленькая, дайте дефолт
    if not np.isfinite([xmin, xmax, ymin, ymax]).all() or (xmax - xmin < 1e-6) or (ymax - ymin < 1e-6):
        xmin, xmax, ymin, ymax = -3, 3, -3, 3

    xs = np.linspace(xmin, xmax, 400)
    ys = np.linspace(ymin, ymax, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]], dtype=float))

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    levels = np.linspace(np.nanmin(Z), np.nanpercentile(Z, 95), 25)
    ax.contour(X, Y, Z, levels=levels)
    ax.plot(traj[:, 0], traj[:, 1], marker='o', markersize=2, linewidth=1, label="trajectory")
    ax.scatter([traj[0, 0]], [traj[0, 1]], s=60, marker='s', label="start")
    ax.scatter([traj[-1, 0]], [traj[-1, 1]], s=60, marker='*', label="end")

    if minima_hint:
        ax.set_title(f"{title}\n{minima_hint}")
    else:
        ax.set_title(title)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    plt.show()


# =========================
#  CLI + интерактив
# =========================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Universal GD with Armijo. Registry, custom input, early stop, auto x0.")
    p.add_argument("--func", help="Function key from registry (see --list). If omitted, interactive mode starts.")
    p.add_argument("--list", action="store_true", help="List available functions and exit.")
    p.add_argument("--x0", help="Initial point as comma-separated floats, e.g. '0,0' or '0,0,0'.")
    p.add_argument("--auto-x0", action="store_true", help="Automatically choose a good starting point.")
    p.add_argument("--auto-x0-samples", type=int, default=64, help="Number of random candidates for auto-x0.")
    p.add_argument("--auto-x0-radius", type=float, default=5.0, help="Radius of sampling cube for auto-x0.")
    p.add_argument("--auto-x0-seed", type=int, default=42, help="Random seed for auto-x0.")
    p.add_argument("--step", type=float, default=1.0, help="Initial step.")
    p.add_argument("--no-backtracking", action="store_true", help="Disable Armijo backtracking (use fixed step).")
    p.add_argument("--grad-policy", choices=["auto", "jax", "finite"], default="auto", help="How to build gradient.")
    p.add_argument("--fd-eps", type=float, default=1e-8, help="Finite-diff base step (central difference).")
    p.add_argument("--tol-grad", type=float, default=1e-10, help="Stop when ||grad|| < tol_grad.")
    p.add_argument("--tol-step", type=float, default=1e-14, help="Stop when ||Δx|| < tol_step.")
    p.add_argument("--tol-f", type=float, default=1e-12, help="Early stop if |Δf| < tol_f over patience_f iters.")
    p.add_argument("--patience-f", type=int, default=500, help="Window (iters) for f-stabilization.")
    p.add_argument("--max-iters", type=int, default=100000, help="Max iterations.")
    p.add_argument("--log-every", type=int, default=500, help="Log every N iterations (0 to disable).")
    p.add_argument("--compare-scipy", action="store_true", help="Also run scipy.optimize.minimize for comparison.")
    p.add_argument("--interactive", action="store_true", help="Force interactive mode.")
    return p.parse_args(argv)


def _explain_and_print(label: str, x_star: np.ndarray, f_star: float, iters: int, meta: dict,
                       spec: Optional[FuncSpec]) -> None:
    """
    Красивый человекочитаемый отчет.
    """
    reason_map = {
        "grad_tol": "достигнут порог по норме градиента (||∇f|| < tol_grad)",
        "step_tol": "шаг стал слишком мал (||Δx|| < tol_step)",
        "f_stabilized": "значение функции стабилизировалось (|Δf| ≤ tol_f в окне)",
        "max_iter": "достигнут лимит итераций (max_iter)",
    }
    print("\n— РЕЗУЛЬТАТ —")
    print(f"  Функция:           {label}")
    if spec and spec.minima_note:
        print(f"  Подсказка по минимуму: {spec.minima_note}")
    print(f"  Найденная точка x*: {x_star}")
    print(f"  Значение f(x*):     {f_star:.12e}")
    print(f"  Итераций:           {iters}")
    print(f"  Причина остановки:  {meta.get('stop_reason')} — {reason_map.get(meta.get('stop_reason', ''), '')}")
    print(f"  ||∇f(x*)||:         {meta.get('grad_norm'):.3e}")
    print(f"  Последний шаг α:    {meta.get('alpha'):.3e}")
    print(f"  Backtracking редукций на последнем шаге: {meta.get('bt_reductions')}")
    print("  Пояснение полей:")
    print("    • x* — найденная точка минимума/стационарная точка.")
    print("    • f(x*) — значение функции в найденной точке.")
    print("    • ||∇f|| — насколько «круто» меняется f (ноль на точном минимуме).")
    print("    • α — фактическая длина шага на последней итерации (после бэктрекинга).")
    print("    • backtracking редукций — сколько раз уменьшали шаг, чтобы выполнить условие Армихо.\n")


def run_gd(
        f: Callable[[NDArray[np.floating]], float],
        x0: NDArray[np.floating],
        args: argparse.Namespace,
        label: str,
        spec: Optional[FuncSpec] = None
) -> None:
    gd = GradientDescent(
        f=f,
        g=None,
        step=args.step,
        tol_grad=args.tol_grad,
        tol_step=args.tol_step,
        tol_f=args.tol_f,
        patience_f=args.patience_f,
        max_iter=args.max_iters,
        use_backtracking=(not args.no_backtracking),
        grad_policy=args.grad_policy,
        fd_eps=args.fd_eps,
        log_every=args.log_every,
    )
    x_star, f_star, iters, hist_f, meta, traj = gd.run(x0)
    print(f"[GD] {label} -> x={x_star}, f={f_star:.12e}, iters={iters}, stop={meta['stop_reason']}, "
          f"||g||={meta['grad_norm']:.3e}, alpha={meta['alpha']:.2e}, bt_red={meta['bt_reductions']}")
    _explain_and_print(label, x_star, f_star, iters, meta, spec)

    if args.compare_scipy:
        if not _HAS_SCIPY:
            print("[WARN] SciPy is not installed; skipping comparison.")
        else:
            res = minimize(fun=f, x0=np.array(x0, dtype=float))
            print(f"[SciPy] {label} -> x={res.x}, f={res.fun:.12e}, iters={getattr(res, 'nit', None)}\n")

    # интерактивное предложение визуализации для 2D
    dim = x0.size
    if dim == 2:
        ans = input("Построить 2D визуализацию траектории? [y/N]: ").strip().lower()
        if ans == "y":
            minima_hint = spec.minima_note if spec else None
            visualize_2d_contours_and_path(f, traj, minima_hint=minima_hint, title=f"{label} (2D)")


def interactive_flow(args: argparse.Namespace) -> int:
    print("=== Universal Gradient Descent (interactive) ===")
    print("Выбери режим:")
    print("  1) Функция из реестра")
    print("  2) Ввести свою функцию вручную")
    choice = input("Твой выбор [1/2]: ").strip() or "1"

    if choice == "1":
        print("\nДоступные функции:\n" + list_functions())
        key = input("Введи ключ функции (например, rosenbrock): ").strip()
        if key not in FUNC_REGISTRY:
            print(f"[ERROR] Неизвестная функция '{key}'.")
            return 2
        spec = FUNC_REGISTRY[key]

        if spec.dim is None:
            x0_str = input("Введи начальную точку x0 (например '0,0,0'): ").strip()
            dim = len(x0_str.split(",")) if x0_str else 0
        else:
            dim = spec.dim
            use_auto = input(f"Автовыбрать x0 размерности {dim}? [y/N]: ").strip().lower() == "y"
            if use_auto:
                x0 = autoselect_x0(spec.func, dim, samples=args.auto_x0_samples, radius=args.auto_x0_radius,
                                   seed=args.auto_x0_seed)
                print(f"[auto-x0] выбран старт: x0={x0}")
                run_gd(spec.func, x0, args, label=spec.name, spec=spec)
                return 0
            x0_str = input(f"Введи x0 размерности {dim} (например '0,0'): ").strip()

        x0 = np.array([float(s) for s in x0_str.split(",")], dtype=float)
        if (spec.dim is not None) and (x0.size != spec.dim):
            print(f"[ERROR] Ожидалась размерность {spec.dim}, а получено {x0.size}.")
            return 2

        run_gd(spec.func, x0, args, label=spec.name, spec=spec)
        return 0

    elif choice == "2":
        dim = int(input("Введи размерность / количество переменных функции (целое > 0): ").strip())
        expr = input(
            "Введи выражение для f(x):\n"
            "  · можно использовать x, x[i] (векторный стиль), np.*, sin, cos, exp, log, sqrt, abs, pi, e\n"
            "  · или имена x0,x1,... (доступны также y≡x1, z≡x2)\n"
            "Пример 2D:  2*x0**2 - 4*x0*x1 + x1**4 + 2\n"
            "Твоя f(x) = "
        )
        f = build_custom_function(expr, dim)

        use_auto = input(f"Автовыбрать x0 размерности {dim}? [y/N]: ").strip().lower() == "y"
        if use_auto:
            x0 = autoselect_x0(f, dim, samples=args.auto_x0_samples, radius=args.auto_x0_radius, seed=args.auto_x0_seed)
            print(f"[auto-x0] выбран старт: x0={x0}")
            run_gd(f, x0, args, label="custom", spec=None)
            return 0

        x0_str = input(f"Введи начальную точку x0 через запятую из {dim} чисел (например '0,0'): ").strip()
        x0 = np.array([float(s) for s in x0_str.split(",")], dtype=float)
        if x0.size != dim:
            print(f"[ERROR] Ожидалась размерность {dim}, а получено {x0.size}.")
            return 2

        run_gd(f, x0, args, label="custom", spec=None)
        return 0

    else:
        print("[ERROR] Неверный выбор.")
        return 2


def parse_args_top(argv: Optional[List[str]] = None) -> argparse.Namespace:
    # Оставлено для обратной совместимости с твоими call-сценариями
    return parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args_top(argv)

    if args.list:
        print("Available functions:\n" + list_functions())
        return 0

    if args.interactive or (args.func is None):
        return interactive_flow(args)

    key = args.func
    if key not in FUNC_REGISTRY:
        print(f"[ERROR] Unknown --func '{key}'. Use --list to see available.", file=sys.stderr)
        return 2

    spec = FUNC_REGISTRY[key]
    dim = spec.dim if spec.dim is not None else None

    # авто x0 при необходимости
    if args.auto_x0:
        if dim is None:
            if args.x0 is None:
                print("[ERROR] For dim=any functions with --auto-x0, please supply --x0 to define dimension.",
                      file=sys.stderr)
                return 2
            dim = len(args.x0.split(","))
        x0 = autoselect_x0(spec.func, dim, samples=args.auto_x0_samples, radius=args.auto_x0_radius,
                           seed=args.auto_x0_seed)
    else:
        if args.x0 is None:
            print("[ERROR] --x0 is required in non-interactive mode (or use --auto-x0).", file=sys.stderr)
            return 2
        try:
            x0 = np.array([float(s) for s in args.x0.split(",")], dtype=float)
        except Exception:
            print("[ERROR] --x0 must be comma-separated floats.", file=sys.stderr)
            return 2

    if (spec.dim is not None) and (x0.size != spec.dim):
        print(f"[ERROR] Function '{key}' expects dim={spec.dim}, but x0 has dim={x0.size}.", file=sys.stderr)
        return 2
    if (spec.dim is None) and (x0.size < 1):
        print(f"[ERROR] Function '{key}' expects dim>=1.", file=sys.stderr)
        return 2
    if key == "rosenbrock_nd" and x0.size < 2:
        print("[ERROR] rosenbrock_nd requires n>=2.", file=sys.stderr)
        return 2

    run_gd(spec.func, x0, args, label=spec.name, spec=spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
