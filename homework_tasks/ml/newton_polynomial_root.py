"""
Модуль: Метод Ньютона (Ньютона–Рафсона) для поиска корня уравнения f(x)=0.

Назначение
----------
Дает универсальную реализацию метода Ньютона для произвольной пары функций
(f, f'), а также конкретное решение учебной задачи:
    f(x) = 6x^5 - 5x^4 - 4x^3 + 3x^2
с начальной точкой x0 = 0.7.

Особенности реализации
----------------------
1) ООП-обертка NewtonRaphson с конфигом:
    - tol: требуемая точность по приращению аргумента,
    - max_iter: ограничение итераций,
    - eps_df: минимально допустимый модуль производной,
    - damping: опциональное демпфирование шага (линейный поиск по половинному шагу),
     чтобы избежать дивергенции, если очередной шаг ухудшает значение |f(x)|.
2) Подробные docstring'и и типизация для удобства поддержки.
3) Возвращается не только корень, но и метаданные прогона (число итераций и флаг сходимости).

Математическая справка
----------------------
Итерация Ньютона:
    x_{k+1} = x_k - f(x_k) / f'(x_k)

Сходимость, как правило, квадратичная при:
    - f непрерывно дифференцируема в окрестности корня,
    - f'(x*) != 0,
    - начальное приближение x0 достаточно близко к корню x*.

Потенциальные проблемы
----------------------
    - f'(x_k) ≈ 0 -> деление на крошечные числа -> огромный шаг; мы перехватываем случай,
      если |f'(x_k)| < eps_df, и пытаемся смягчить шаг демпфированием.
    - Далекая стартовая точка -> метод может "улететь"; демпфирование и ограничение итераций помогают.

Как запустить
-------------
    poetry run python mephi_homework_tasks/homework_tasks/linear_algebra/newton_polynomial_root.py

Ожидаемый результат для задачи из учебника:
    Root ~ 0.6286669787764609
    Answer (rounded to 3 decimals): 0.629
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional


@dataclass(frozen=True)
class NewtonConfig:
    """
    Параметры метода Ньютона.
    """
    tol: float = 1e-12  # критерий остановки по |x_{k+1} - x_k|
    max_iter: int = 100  # защита от бесконечного цикла
    eps_df: float = 1e-12  # минимально допустимый модуль производной
    damping: bool = True  # включить демпфирование (половиним шаг, если не улучшает |f|)


class NewtonRaphson:
    """
    Универсальный решатель уравнения f(x)=0 методом Ньютона.

    Пример:
        solver = NewtonRaphson(f, df, NewtonConfig())
        root, info = solver.solve(x0=0.7)
    """

    def __init__(self, f: Callable[[float], float], df: Callable[[float], float],
                 config: Optional[NewtonConfig] = None):
        self.f = f
        self.df = df
        self.cfg = config or NewtonConfig()

    def solve(self, x0: float) -> Tuple[float, Dict[str, object]]:
        """
        Запускает метод Ньютона.

        Args:
            x0: стартовая точка.

        Returns:
            (root, info)
            root: найденное значение корня
            info: словарь с метаданными:
                  - 'iterations': число итераций,
                  - 'converged': bool-флаг сходимости,
                  - 'last_f': значение f в найденной точке,
                  - 'message': текстовое описание итога.
        """
        x = float(x0)
        fx = self.f(x)

        for k in range(1, self.cfg.max_iter + 1):
            dfx = self.df(x)

            if abs(dfx) < self.cfg.eps_df:
                # Производная слишком мала — делить опасно.
                # Попробуем только демпфировать шаг по направлению градиента функции.
                msg = "Derivative too small; applying damping" if self.cfg.damping else "Derivative too small"
                if not self.cfg.damping:
                    return x, {"iterations": k - 1, "converged": False, "last_f": fx, "message": msg}

            # Базовый шаг Ньютона
            step = fx / dfx if abs(dfx) >= self.cfg.eps_df else 0.0
            x_new = x - step

            # При необходимости — демпфирование (линейный поиск по половинному шагу)
            if self.cfg.damping:
                f_prev_abs = abs(fx)
                lambda_factor = 1.0
                # уменьшаем шаг, пока не улучшим |f|
                while True:
                    f_new = self.f(x_new)
                    if abs(f_new) <= f_prev_abs or lambda_factor < 1e-6:
                        break
                    lambda_factor *= 0.5
                    x_new = x - lambda_factor * step
                fx_new = f_new
            else:
                fx_new = self.f(x_new)

            # Критерий остановки по аргументу
            if abs(x_new - x) < self.cfg.tol:
                return x_new, {
                    "iterations": k,
                    "converged": True,
                    "last_f": fx_new,
                    "message": "Converged by |Δx| < tol",
                }

            # Переход к новой точке
            x, fx = x_new, fx_new

        # Если дошли сюда — не сошлось за max_iter
        return x, {
            "iterations": self.cfg.max_iter,
            "converged": False,
            "last_f": fx,
            "message": "Reached max_iter without satisfying tolerance",
        }


# ---------- Конкретная учебная задача ----------

def f_polynomial(x: float) -> float:
    """f(x) = 6x^5 - 5x^4 - 4x^3 + 3x^2"""
    return (
            6.0 * x ** 5
            - 5.0 * x ** 4
            - 4.0 * x ** 3
            + 3.0 * x ** 2
    )


def df_polynomial(x: float) -> float:
    """f'(x) = 30x^4 - 20x^3 - 12x^2 + 6x"""
    return (
            30.0 * x ** 4
            - 20.0 * x ** 3
            - 12.0 * x ** 2
            + 6.0 * x
    )


if __name__ == "__main__":
    # Стартовое приближение из условия
    x0 = 0.7

    # Конфигурация метода Ньютона (строгая точность; демпфирование включено)
    cfg = NewtonConfig(tol=1e-12, max_iter=100, eps_df=1e-12, damping=True)

    solver = NewtonRaphson(f_polynomial, df_polynomial, cfg)
    root, info = solver.solve(x0)

    print(f"Root ~ {root:.16f}")
    print(f"Converged: {info['converged']} in {info['iterations']} iterations; f(root) = {info['last_f']:.3e}")
    print(f"Answer (rounded to 3 decimals): {root:.3f}")
