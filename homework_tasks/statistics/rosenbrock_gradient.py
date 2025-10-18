"""
Градиентный спуск для функции Розенброка с бэктрек-линейным поиском (Armijo).
- Дает эталонную реализацию градиентного спуска (GD) с выбором шага по правилу
      Армихо (backtracking line search).
- Показывает, как подключить аналитический градиент и как сравнить с SciPy BFGS.

ЦЕЛЕВАЯ ФУНКЦИЯ
---------------
Используем классическую "банановую" функцию Розенброка в R^2:
    f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2

Ее градиент:
    ∂f/∂x = -2 + 2x - 400x(y - x^2)
    ∂f/∂y = 200(y - x^2)

Глобальный минимум: (x*, y*) = (1, 1).

АЛГОРИТМ
--------
Итерация градиентного спуска:
    x_{k+1} = x_k - α_k * ∇f(x_k),
где α_k выбираем бэктрекингом (Armijo):
    f(x + α p) ≤ f(x) + c * α * ⟨∇f(x), p⟩, p = -∇f(x),
с параметрами по умолчанию c=1e-4, rho=0.5.

КРИТЕРИИ ОСТАНОВКИ
------------------
1) ||∇f(x_k)||_2 < tol_grad
2) ||x_{k+1} - x_k||_2 < tol_step
3) Достигнут max_iter

ПРАКТИЧЕСКИЕ СОВЕТЫ
-------------------
- Для "узких долин" (Rosenbrock) фиксированный шаг зачастую нестабилен.
  Линейный поиск резко улучшает сходимость.
- Не печатайте лог каждую итерацию: это тормозит. Логируйте редко (log_every).
- Проверьте, что f и ∇f возвращают конечные значения — иначе прекращайте (diverge guard).

ПРИМЕР
------
>>> import numpy as np
>>> x0 = (0.0, 0.0)
>>> gd = GradientDescent(f=rosenbrock, g=rosenbrock_grad, step=1.0, use_backtracking=True)
>>> x_star, f_star, iters, _ = gd.run(x0)
>>> np.allclose(x_star, np.array([1.0, 1.0]), atol=1e-4)
True

СРАВНЕНИЕ СО SCIPY
------------------
В конце модуля, в блоке __main__, показан запуск:
    - "ручного рассчета" GD,
    - SciPy minimize(method="BFGS") с аналитическим градиентом.

ЗАВИСИМОСТИ
-----------
numpy, scipy (для сравнения), dataclasses (встроенный), typing.
"""

from dataclasses import dataclass
from typing import Callable, Tuple, List
import numpy as np
from numpy.typing import NDArray
from math import isfinite
from scipy.optimize import minimize


# =========================
#  ЦЕЛЕВАЯ ФУНКЦИЯ И ГРАДИЕНТ
# =========================

def rosenbrock(xy: NDArray[np.floating]) -> float:
    """
    Значение функции Розенброка.

    Параметры
    ---------
    xy : np.ndarray shape (2,)
        Точка (x, y).

    Возвращает
    ----------
    float
        Значение f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2.

    Замечания
    ---------
    - Функция имеет узкую долину, ведущую к глобальному минимуму в (1, 1).
    - Чувствительна к выбору шага в простом GD без line search.
    """
    x, y = float(xy[0]), float(xy[1])
    return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2


def rosenbrock_grad(xy: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Аналитический градиент функции Розенброка.

    Параметры
    ---------
    xy : np.ndarray shape (2,)
        Точка (x, y).

    Возвращает
    ----------
    np.ndarray shape (2,)
        Вектор градиента [df/dx, df/dy].

    Формулы
    -------
        df/dx = -2 + 2x - 400x(y - x^2)
        df/dy = 200(y - x^2)
    """
    x, y = float(xy[0]), float(xy[1])
    dx = -2.0 + 2.0 * x - 400.0 * x * (y - x ** 2)
    dy = 200.0 * (y - x ** 2)
    return np.array([dx, dy], dtype=float)


# =========================
#  ЛИНЕЙНЫЙ ПОИСК (ARMIJO)
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
) -> float:
    """
    Подбор шага по правилу Армихо (backtracking).

    Ищем наименьшее α из геометрической прогрессии {α0, α0*rho, α0*rho^2, ...},
    удовлетворяющее условию достаточного убывания:
        f(x + α p) ≤ f(x) + c * α * ⟨∇f(x), p⟩,
    где p — направление спуска (обычно p = -∇f(x)), 0 < c < 1, 0 < rho < 1.

    Параметры
    ---------
    f, g : callable
        Целевая функция и ее градиент.
    x : np.ndarray
        Текущая точка.
    p : np.ndarray
        Направление (обычно -grad).
    alpha0 : float, optional
        Стартовый шаг (по умолчанию 1.0).
    c : float, optional
        Константа Армихо (по умолчанию 1e-4).
    rho : float, optional
        Множитель уменьшения шага (по умолчанию 0.5).
    max_trials : int, optional
        Максимальное число уменьшений шага.

    Возвращает
    ----------
    float
        Найденный шаг α (возможно сильно уменьшенный).

    Замечания
    ---------
    - При слишком узкой долине алгоритм может выполнять несколько уменьшений,
      это нормально.
    - Если условие не выполнено за max_trials попыток, возвращаем текущее α,
      чтобы не зациклиться (хотя оно может быть "неидеальным").
    """
    fx = f(x)
    gx = g(x)
    dot = float(np.dot(gx, p))
    alpha = alpha0
    for _ in range(max_trials):
        if f(x + alpha * p) <= fx + c * alpha * dot:
            return alpha
        alpha *= rho
    return alpha


# =========================
#  КЛАСС ГРАДИЕНТНОГО СПУСКА
# =========================

@dataclass(slots=True)
class GradientDescent:
    """
    Классическая схема градиентного спуска с опциональным backtracking line search.

    Аргументы конструктора
    ----------------------
    f : callable
        Целевая функция f(x) -> float.
    g : callable
        Градиент ∇f(x) -> np.ndarray.
    step : float, optional
        Стартовый шаг для линейного поиска (или фиксированный шаг, если use_backtracking=False).
        По умолчанию 1.0.
    tol_grad : float, optional
        Порог по норме градиента (||∇f||_2 < tol_grad) — критерий остановки.
        По умолчанию 1e-8.
    tol_step : float, optional
        Порог по длине шага (||x_{k+1} - x_k||_2 < tol_step) — дополнительный стоп.
        По умолчанию 1e-12.
    max_iter : int, optional
        Максимальное число итераций. По умолчанию 100_000.
    use_backtracking : bool, optional
        Включить/выключить backtracking. По умолчанию True.
    log_every : int, optional
        Периодичность лога (каждые N итераций). По умолчанию 500.

    Методы
    ------
    run(x0) -> (x_star, f_star, iters, history)
        Запускает оптимизацию из точки x0. Возвращает найденную точку,
        значение функции, число итераций и историю f по итерациям.

    Пример
    ------
    >>> gd = GradientDescent(rosenbrock, rosenbrock_grad, step=1.0, use_backtracking=True)
    >>> x_star, f_star, iters, hist = gd.run((0.0, 0.0))
    >>> x_star
    array([1.000..., 1.000...])
    """
    f: Callable[[NDArray[np.floating]], float]
    g: Callable[[NDArray[np.floating]], NDArray[np.floating]]
    step: float = 1.0
    tol_grad: float = 1e-8
    tol_step: float = 1e-12
    max_iter: int = 100_000
    use_backtracking: bool = True
    log_every: int = 500

    def run(self, x0: Tuple[float, float]) -> Tuple[NDArray[np.floating], float, int, List[float]]:
        """
        Запуск градиентного спуска.

        Параметры
        ---------
        x0 : tuple[float, float]
            Начальная точка (x, y).

        Возвращает
        ----------
        (x_star, f_star, iters, history)
            x_star : np.ndarray shape (2,)
                Найденная точка (приближение к минимуму).
            f_star : float
                Значение f в найденной точке.
            iters : int
                Количество выполненных итераций.
            history : list[float]
                История значений f по итерациям (для анализа сходимости).

        Исключения
        ----------
        FloatingPointError
            Если f или ∇f вернули нечисловые/бесконечные значения (признак расходимости).

        Замечания
        ---------
        - Если use_backtracking=True, шаг подбирается по правилу Армихо.
        - Если False — используется фиксированный шаг `step`.
        - Алгоритм останавливается по ||∇f||_2, длине шага и/или max_iter.
        """
        x = np.asarray(x0, dtype=float)
        history: List[float] = []

        for k in range(1, self.max_iter + 1):
            gk = self.g(x)
            ng = float(np.linalg.norm(gk))
            fk = self.f(x)
            history.append(fk)

            if not isfinite(fk) or not np.isfinite(gk).all():
                raise FloatingPointError("f/grad not finite; diverged.")

            if ng < self.tol_grad:
                break

            p = -gk  # направление спуска
            alpha = (
                backtracking_armijo(self.f, self.g, x, p, alpha0=self.step)
                if self.use_backtracking else self.step
            )
            x_new = x + alpha * p

            if np.linalg.norm(x_new - x) < self.tol_step:
                x = x_new
                break

            x = x_new

            if self.log_every and (k % self.log_every == 0):
                print(f"[{k}] f={fk:.6e}, ||g||={ng:.3e}, alpha={alpha:.2e}, x={x}")

        return x, float(self.f(x)), k, history


# =========================
#  ДЕМО: запуск из командной строки
# =========================

if __name__ == "__main__":
    # Стартовая точка как в твоих примерах
    x0 = (0.0, 0.0)

    # 1) Наш градиентный спуск с бэктрек-линейным поиском
    gd = GradientDescent(
        f=rosenbrock,
        g=rosenbrock_grad,
        step=1.0,  # старт для backtracking
        use_backtracking=True,  # включаем линийный поиск
        tol_grad=1e-10,
        tol_step=1e-14,
        log_every=500
    )
    x_star, f_star, iters, hist = gd.run(x0)
    print(f"GD result:  x={x_star}, f={f_star:.6e}, iters={iters}")

    # 2) SciPy minimize (BFGS) с аналитическим градиентом — для сравнения
    res = minimize(
        fun=rosenbrock,
        x0=np.array(x0, dtype=float),
        jac=rosenbrock_grad,
        method="BFGS",
        options={"gtol": 1e-10}
    )
    print(f"BFGS result: x={res.x}, f={res.fun:.6e}, iters={res.nit}")
