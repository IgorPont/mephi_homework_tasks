"""
Модуль демонстрирует обучение многомерной линейной регрессии методом
градиентного спуска на датасете Advertising (признаки: TV, radio, newspaper),
а также визуализацию сходимости MSE и вычисление метрик качества (MSE, R²)
с подробными пояснениями.

Результаты включают:
    - оценку коэффициентов модели,
    - финальные метрики MSE и R²,
    - интерпретацию коэффициентов и сходимости,
    - печать пояснений «человеческим языком».

Ожидаемые коэффициенты на оригинальных данных:
[b0, TV, Radio, Newspaper] ≈ [4.635, 0.054, 0.107, -0.002]
"""

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================
# Конфигурация обучения
# ==========================

@dataclass
class GDConfig:
    """Параметры градиентного спуска."""
    # Шаг обучения 5 * 10 ^ −5 = 0.00005
    learning_rate: float = 5e-5
    # Количество итераций шагов
    n_iter: int = 500_000
    # Как часто модель записывает значение функции ошибки (MSE) во время обучения
    record_every: int = 1_000


# ==========================
# Модель линейной регрессии
# ==========================

class GradientDescentLinearRegressor:
    """Линейная регрессия на чистом NumPy с обучением градиентным спуском."""

    def __init__(self, config: Optional[GDConfig] = None):
        self.config = config or GDConfig()
        self.theta_: Optional[np.ndarray] = None
        self.costs_: List[float] = []
        self.steps_: List[int] = []

    @staticmethod
    def add_intercept(X: np.ndarray) -> np.ndarray:
        """Добавляет столбец единиц (для коэффициента b0)."""
        n = X.shape[0]
        return np.append(np.ones((n, 1)), X, axis=1)

    @staticmethod
    def _compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Среднеквадратичная ошибка (MSE) в виде (1/n) * ||Xθ - y||^2.
        Константы 1/2 нет — компенсируется подбором learning_rate.
        """
        n = y.shape[0]
        residuals = X @ theta - y
        return float((residuals ** 2).sum() / n)

    @staticmethod
    def _compute_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Градиент MSE по θ: (Xᵀ (Xθ - y)) / n.
        Коэффициент 2 поглощён learning_rate.
        """
        n = y.shape[0]
        residuals = X @ theta - y
        return (X.T @ residuals) / n

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientDescentLinearRegressor":
        """Обучает модель градиентным спуском."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X_ext = self.add_intercept(X)
        theta = np.zeros((X_ext.shape[1], 1))

        lr = self.config.learning_rate
        n_iter = self.config.n_iter
        record_every = max(1, int(self.config.record_every))

        self.costs_.clear()
        self.steps_.clear()

        for i in range(n_iter):
            grad = self._compute_gradient(X_ext, y, theta)
            theta -= lr * grad
            if i % record_every == 0:
                cost = self._compute_cost(X_ext, y, theta)
                self.costs_.append(cost)
                self.steps_.append(i)

        self.theta_ = theta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает предсказания ŷ."""
        if self.theta_ is None:
            raise RuntimeError("Модель не обучена. Вызови fit(X, y).")
        X_ext = self.add_intercept(X)
        return X_ext @ self.theta_

    # --- Метрики ---

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Коэффициент детерминации R² = 1 - SS_res / SS_tot,
        где SS_res = ∑(y - ŷ)², SS_tot = ∑(y - ȳ)².
        """
        if y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = y_true.ravel()
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # --- Визуализация ---

    def plot_convergence(self) -> None:
        """Рисует график сходимости функции стоимости (MSE)."""
        if not self.costs_:
            raise RuntimeError("Нет истории обучения. Сначала вызови .fit().")
        plt.figure(figsize=(8, 5))
        plt.plot(self.steps_, self.costs_, linewidth=2)
        plt.title("Сходимость функции стоимости (MSE)")
        plt.xlabel("Итерации")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_convergence_multi(
            histories: Iterable[Tuple[str, List[int], List[float]]]
    ) -> None:
        """Сравнивает сходимость при разных значениях alpha."""
        plt.figure(figsize=(9, 5))
        for label, steps, costs in histories:
            plt.plot(steps, costs, label=label, linewidth=2)
        plt.title("Сходимость MSE при разных alpha")
        plt.xlabel("Итерации")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ==========================
# Основной запуск
# ==========================

if __name__ == "__main__":
    # --- Определяем путь к CSV ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "input", "Advertising.csv")

    # --- Загружаем данные ---
    df = pd.read_csv(CSV_PATH, index_col=0)
    X = df[["TV", "radio", "newspaper"]].values
    y = df["sales"].values.reshape(-1, 1)

    # --- Обучаем модель ---
    config = GDConfig(learning_rate=0.00005, n_iter=500_000, record_every=1_000)
    model = GradientDescentLinearRegressor(config=config).fit(X, y)

    # --- Предсказания и метрики ---
    y_pred = model.predict(X)
    final_mse = float(np.mean((y_pred - y) ** 2))
    r2 = model.r2_score(y, y_pred)

    # --- Вывод результатов + подробные пояснения ---
    theta = model.theta_.flatten()

    print("\n=== РЕЗУЛЬТАТ ОБУЧЕНИЯ МОДЕЛИ ===\n")

    print("Коэффициенты линейной регрессии (интерпретация ceteris paribus):")
    print(f"  b0 (свободный член): {theta[0]:.3f}")
    print("     └ Базовый уровень продаж при нулевых затратах на все каналы рекламы.")
    print(f"  b1 (TV):             {theta[1]:.3f}")
    print("     └ При увеличении расходов на TV на 1 единицу продажи растут примерно на b1 единиц,")
    print("       если radio и newspaper фиксированы (ceteris paribus).")
    print(f"  b2 (Radio):          {theta[2]:.3f}")
    print("     └ Аналогично для радио: вклад в рост продаж при неизменности других каналов.")
    print(f"  b3 (Newspaper):      {theta[3]:.3f}")
    print("     └ Если близок к нулю (или отрицателен), влияние рекламных объявлений в газетах")
    print("       статистически мало или смещения данных не позволяют выявить явный эффект.\n")

    print(f"Финальная MSE (среднеквадратичная ошибка): {final_mse:.5f}")
    print("  └ Показывает средний квадрат отклонения предсказаний от истинных продаж.")
    print("    Меньше — лучше (но сравнивать корректно на одной и той же шкале и выборке).")
    print(f"R² (коэффициент детерминации): {r2:.4f}")
    print(
        "  └ Доля дисперсии продаж, объяснённая моделью (1.0 — идеальная подгонка, 0.0 — нет улучшения относительно среднего).")
    print("    Для одного признака TV R² ≈ 0.61 (как в классическом примере), для трёх признаков обычно выше.\n")

    print("Сходимость (Convergence):")
    print("  └ Во время обучения значение MSE должно монотонно убывать и выходить на плато.")
    print("    Если кривая MSE «пилит» или растёт — уменьшите learning_rate (alpha).")
    print(f"Параметры обучения: alpha = {config.learning_rate}, итераций = {config.n_iter}\n")

    # --- Визуализируем сходимость ---
    model.plot_convergence()

    # --- (Дополнительно) сравнение разных alpha ---
    # histories = []
    # for alpha in (2e-5, 5e-5, 1e-4):
    #     cfg = GDConfig(learning_rate=alpha, n_iter=150_000, record_every=1_000)
    #     m = GradientDescentLinearRegressor(cfg).fit(X, y)
    #     histories.append((f"alpha={alpha:g}", m.steps_, m.costs_))
    # GradientDescentLinearRegressor.plot_convergence_multi(histories)
