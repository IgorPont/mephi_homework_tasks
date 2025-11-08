"""
Условие задания № 3:
    Вам предложены данные с информацией о успеваемости студентов:
    Независимые переменные:
        - Hours Studied: Общее количество часов, потраченных на учебу каждым студентом.
        - Previous Scores: Баллы, полученные студентами на предыдущих экзаменах.
        - Sleep Hours: Среднее количество часов сна студента в сутки.
        - Sample Question Papers Practiced: Количество пробных экзаменационных работ, с которыми студент занимался.
    Целевая переменная:
        Performance Index: Показатель общей успеваемости каждого студента.
        Индекс успеваемости отражает академическую успеваемость студента и округляется до ближайшего целого числа.
        Индекс варьируется от 10 до 100, при этом более высокие значения свидетельствуют о более высокой успеваемости.
    Решите задачу линейной регрессии, реализовав градиентный спуск самостоятельно,
    не используя готовое решение из какой-либо библиотеки.

Размещение входных данных:
    mephi_homework_tasks/sessions_tasks/math_analysis/homework_tasks_1/input/dataset_for_task_3.txt

Пример результата вывода:
    --- Собственная реализация GD - TRAIN ---
    MSE : 4.169736
    RMSE: 2.041993
    MAE : 1.619299
    R^2 : 0.988690

    --- Собственная реализация GD - TEST  ---
    MSE : 4.082580
    RMSE: 2.020539
    MAE : 1.611106
    R^2 : 0.988983

    --- sklearn.SGDRegressor (градиентный спуск) - TRAIN ---
    MSE : 4.251002
    RMSE: 2.061796
    MAE : 1.635593
    R^2 : 0.988469

    --- sklearn.SGDRegressor (градиентный спуск) - TEST  ---
    MSE : 4.154529
    RMSE: 2.038266
    MAE : 1.620790
    R^2 : 0.988789

    --- sklearn.LinearRegression (референс) - TRAIN ---
    MSE : 4.169736
    RMSE: 2.041993
    MAE : 1.619305
    R^2 : 0.988690

    --- sklearn.LinearRegression (референс) - TEST  ---
    MSE : 4.082628
    RMSE: 2.020552
    MAE : 1.611121
    R^2 : 0.988983

    --- Коэффициенты (scaled space) ---
    GD  : bias=55.310847, weights=[ 7.401225 17.637049  0.304296  0.810029  0.548878]
    SGD : bias=55.295520, weights=[ 7.576335 17.651592  0.245981  0.678228  0.719938]
    OLS : bias=55.311500, weights=[ 7.401341 17.637271  0.304291  0.810031  0.548842]

    === Итог ===
    1) Реализован батчевый градиентный спуск (с нуля), нотация y_pred - везде.
    2) Проведено сравнение с библиотечным градиентным спуском (SGDRegressor).
    3) Для контроля показаны метрики закрытой формулы (LinearRegression).
    4) Результаты на тесте сопоставимы -> "ручная" реализация градиентного спуска корректна.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import SGDRegressor, LinearRegression


# === Реализация с нуля (батч-GD) ===

@dataclass
class GDConfig:
    """
    Конфигурация обучения батчевого градиентного спуска.
    Пояснения:
        - lr: шаг обучения (если слишком большой - расходится; если слишком маленький - долго сходится).
        - max_iter: максимум итераций (проходов по батчу, что равняется всему X).
        - tol: критерий сходимости по изменению MSE между итерациями.
        - record_every: частота логирования прогресса (в итерациях).
        - early_stopping: включать ли раннюю остановку по лучшему MSE (среднеквадратичной ошибке).
        - patience: сколько итераций терпеть отсутствие улучшения перед остановкой (для преждевременной остановки).
        - random_state: фиксатор (seed) генератора случайных чисел, который обеспечит,
            что модель при каждом запуске обучится на тех же самых данных,
            и результаты сравнения будут стабильными.
    """
    lr: float = 1e-2  # 0.01
    max_iter: int = 50_000
    tol: float = 1e-8  # 0.00000001
    record_every: int = 2_000
    early_stopping: bool = True
    patience: int = 5_000
    random_state: int = 42


class LinearRegressionGD:
    """
    Линейная регрессия с обучением методом батчевого градиентного спуска.

    Математическая модель:
        y_pred = X @ w + b

        где:
            - y_pred -> вектор предсказаний модели
            - X -> матрица признаков (N объектов × d признаков)
            - w -> вектор весов (параметры модели)
            - b -> смещение (bias), отвечает за сдвиг прямой
            - y -> вектор реальных значений целевой переменной

    Функция потерь (MSE):
        L(w, b) = (1/N) * ||(X @ w + b) - y||**2 => mse = np.mean((y_pred - y) ** 2)

        где:
            - (y_pred - y) -> вектор ошибок (разница между предсказаниями и реальными значениями)
            - ** 2 -> возводим ошибки в квадрат (чтобы не было отрицательных)
            - np.mean() -> берем среднее значение (суммируем и делим на количество наблюдений N)


    Градиенты (производные функции потерь):
        dL/dw = (2/N) * X^T (Xw + b - y) => (2 / X.shape[0]) * (X.T @ (y_pred - y))
        dL/db = (2/N) * Σ_i ( (x_i ⋅ w + b) - y_i ) => (2 / X.shape[0]) * np.sum(y_pred - y)

        где:
            - X.T -> транспонированная матрица X (размерность d × N)
            - @ -> матричное умножение
            - error -> вектор ошибок (N × 1)
            - результат X.T @ (y_pred - y) -> это вектор dw размерности d × 1
            - np.sum(y_pred - y) - сумма всех ошибок для расчета db.

    Почему выбрал именно batch (батчевый, а не стохастический):
        - В данном случае алгоритм использует весь X на каждой итерации.
        - Легко сопоставимо с теорией и хорошо подходит для небольших датасетов.
    """

    def __init__(self, cfg: GDConfig) -> None:
        """
        Инициализация экземпляра класса линейной регрессии, обучаемой методом градиентного спуска:
            - подготавливает внутреннее состояние;
            - при фиксированном random_state модель гарантированно дает одинаковый результат при каждом запуске,
                что удобно для отладки и сравнения.

        Аргументы:
            cfg (GDConfig):
                Конфигурационный объект, содержащий параметры обучения:
                    - learning_rate (float): шаг градиентного спуска.
                    - n_iterations (int): максимальное количество итераций обучения.
                    - tolerance (float): порог остановки по изменению функции потерь.
                    - random_state (int): фиксатор генератора случайных чисел
                      для воспроизводимости результатов.

        Атрибуты экземпляра:
            self.cfg (GDConfig):
                Сохраняет все настройки обучения для доступа из других методов.

            self.w_ (np.ndarray | None):
                Вектор весов (параметры модели). Инициализируется при первом вызове fit().
                Размерность: (n_features,).

            self.b_ (float | None):
                Смещение (bias) - скалярное значение, добавляемое к предсказанию.

            self.history_ (list[Tuple[int, float]]):
                Список кортежей (итерация, значение функции потерь).
                Используется для анализа сходимости и построения графика изменения MSE.

            self._rng (np.random.Generator):
                Генератор случайных чисел NumPy, инициализированный через random_state.
        """
        self.cfg = cfg
        self.w_: np.ndarray | None = None
        self.b_: float | None = None
        # (iter, mse)
        self.history_: list[Tuple[int, float]] = []

        # Инициализируем генератор для воспроизводимости (при желании можно сделать случайные w)
        self._rng = np.random.default_rng(self.cfg.random_state)

    @staticmethod
    def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Поведение:
            - Вычисляет среднеквадратичную ошибку (MSE) между реальными значениями и предсказаниями модели.
            - Использует матричное умножение для эффективного расчета квадратичных отклонений.
            - Возвращает число типа float (даже при передаче массивов NumPy).
            - Не изменяет входные данные.

        Аргументы:
            y_true (np.ndarray): вектор реальных значений целевой переменной

            y_pred (np.ndarray): вектор предсказанных моделью значений

        Возвращает:
            float: среднеквадратичную ошибку (MSE), измеряющую среднее квадратичное отклонение
                предсказаний от истинных значений. Чем меньше значение, тем точнее модель.
        """
        err = y_pred - y_true
        return float((err @ err) / y_true.shape[0])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Обучение батчевым GD:
            1) Инициализируем параметры (w - нули, b - 0.0).
            2) На каждой итерации считаем y_pred, градиенты, обновляем w и b.
            3) Логируем MSE и проверяем критерии остановки (tol и early_stopping).
        """
        # Разворачиваем размеры матрицы признаков X в две переменные
        # N - количество наблюдений (строк)
        # d - количество признаков (столбцов)
        N, d = X.shape

        # Создаем вектор весов (weights) длиной d, заполненный нулями
        # w_ - параметры линейной модели (по одному коэффициенту на каждый признак)
        # На старте обучения все коэффициенты равны 0, затем корректируются на каждой итерации градиентного спуска
        self.w_ = np.zeros(d, dtype=float)

        # Задаем начальное значение смещения (bias), также равное нулю
        # 0.0 - скаляр, который позволит линии регрессии не проходить через начало координат
        self.b_ = 0.0

        # Инициализируем переменную best_mse (лучшее значение функции потерь из всех итераций) бесконечностью,
        # так как вначале нет ни одного значения, чтобы первое найденное MSE стало лучшим
        best_mse = np.inf

        # Создаем копии текущих параметров модели, соответствующих лучшему состоянию на данный момент
        # best_w и best_b будут хранить веса и смещение, при которых модель показала минимальное значение MSE
        best_w = self.w_.copy()
        best_b = self.b_

        # Счетчик итераций без улучшения метрики
        # На старте равен нулю
        no_improve = 0

        # Сохраняем значение ошибки с предыдущей итерации
        prev_mse = np.inf

        for it in range(1, self.cfg.max_iter + 1):
            # Прямой проход: y_pred = Xw + b
            y_pred = X @ self.w_ + self.b_

            # Градиенты по MSE
            residuals = (y_pred - y)
            grad_w = (2.0 / N) * (X.T @ residuals)
            grad_b = (2.0 / N) * residuals.sum()

            # Шаг GD
            self.w_ = self.w_ - self.cfg.lr * grad_w
            self.b_ = self.b_ - self.cfg.lr * grad_b

            # Текущий MSE
            curr_mse = self._mse(y, X @ self.w_ + self.b_)

            # Запись истории для мониторинга
            if it == 1 or it % self.cfg.record_every == 0:
                self.history_.append((it, curr_mse))

            # Сравниваем текущий MSE и лучший MSE (для ранней остановки по лучшему MSE, если он не обновляется)
            # Плюсуем скаляр 1e-15, чтобы исключить ложное False из-за точности чисел с плавающей точкой
            if curr_mse + 1e-15 < best_mse:
                best_mse = curr_mse
                best_w = self.w_.copy()
                best_b = self.b_
                no_improve = 0
            else:
                no_improve += 1
                if self.cfg.early_stopping and no_improve >= self.cfg.patience:
                    # Возвращаемся к лучшим параметрам
                    self.w_, self.b_ = best_w, best_b
                    break

            # Остановка по относительной стабилизации потерь
            if abs(prev_mse - curr_mse) < self.cfg.tol:
                break
            prev_mse = curr_mse

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает y_pred = X @ w + b.
        Важно: X должен быть в том же масштабе, что и обучающий (после StandardScaler).
        """
        assert self.w_ is not None and self.b_ is not None, "Сначала вызовите fit(X, y)"
        return X @ self.w_ + self.b_


# === Утилиты: метрики и печать ===

def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Сводный отчет по метрикам регрессии
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def print_report(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n--- {title} ---")
    print(f"MSE : {metrics['MSE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE : {metrics['MAE']:.6f}")
    print(f"R^2 : {metrics['R2']:.6f}")


# === Основная логика ===

def main() -> None:
    """
    Пошаговый сценарий:
        1) Загрузка исходного датасета.
        2) Кодирование категориального признака: "Extracurricular Activities" -> {Yes:1, No:0}.
        3) Трейн/Тест разбиение для честной оценки обобщающей способности.
        4) Масштабирование признаков (StandardScaler) - ускоряет и стабилизирует GD.
        5) ОБУЧЕНИЕ собственной реализации LinearRegressionGD (батч-GD):
            - y_pred = X @ w + b
            - градиенты по MSE, обновление параметров
        6) Оценка на train/test: MSE, RMSE, MAE, R^2.
        7) Сравнение с библиотечным градиентным спуском (sklearn.SGDRegressor).
        8) Сравнение с закрытой формулой, то есть точное решение через линейную алгебру (sklearn.LinearRegression).
    """

    # 1) Загрузка данных
    data_path = Path(__file__).parent / "input" / "dataset_for_task_3.txt"
    df = pd.read_csv(data_path)

    # 2) Кодирование категориального признака
    #    Yes/No -> 1/0 - стандартная бинарная кодировка (для удобства работы с моделью)
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

    # Матрица X и целевой вектор y
    feature_names = [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced",
    ]
    target_name = "Performance Index"

    X = df[feature_names].to_numpy(dtype=float)
    y = df[target_name].to_numpy(dtype=float)

    # 3) Разбиение на train/test, чтобы модель не подглядывала в тест, метрики на test -> честная оценка.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Масштабирование признаков
    #    GD чувствителен к масштабу, стандартизация делает ландшафт потерь круглее
    #    и уменьшает разброс шагов по разным осям (fit на train, transform на train/test)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # 5) Обучение собственной реализации (батч-GD)
    #    Параметры подбираем умеренно: lr=5e-3..1e-2 и max_iter до 1e5
    gd_cfg = GDConfig(lr=5e-3, max_iter=120_000, record_every=10_000, patience=20_000)
    model_gd = LinearRegressionGD(gd_cfg)
    model_gd.fit(X_tr_sc, y_tr)

    #    Предсказания на train/test
    y_pred_tr = model_gd.predict(X_tr_sc)  # y_pred (train)
    y_pred_te = model_gd.predict(X_te_sc)  # y_pred (test)

    #    Метрики
    rep_gd_tr = regression_report(y_tr, y_pred_tr)
    rep_gd_te = regression_report(y_te, y_pred_te)

    print_report("Собственная реализация GD - TRAIN", rep_gd_tr)
    print_report("Собственная реализация GD - TEST ", rep_gd_te)

    # 6) Сравнение с библиотечным градиентным спуском (SGDRegressor)
    #    Настраиваем SGD как как можно более близкий к батч-GD:
    #      - loss='squared_error'     -> MSE
    #      - penalty=None             -> без регуляризации, чтобы сопоставить целевые функции
    #      - learning_rate='constant' -> шаг обучения не меняется с каждой итерацией
    #      - max_iter, tol            -> большое число итераций и строгий tol
    sgd = SGDRegressor(
        loss="squared_error",
        penalty=None,
        learning_rate="constant",
        eta0=gd_cfg.lr,
        max_iter=200_000,
        tol=1e-9,
        random_state=gd_cfg.random_state,
        shuffle=True,  # стохастический характер - отличие от батч-GD
        fit_intercept=True,  # b (bias) обучается внутри
        average=False,
    )
    sgd.fit(X_tr_sc, y_tr)
    y_pred_tr_sgd = sgd.predict(X_tr_sc)
    y_pred_te_sgd = sgd.predict(X_te_sc)

    rep_sgd_tr = regression_report(y_tr, y_pred_tr_sgd)
    rep_sgd_te = regression_report(y_te, y_pred_te_sgd)

    print_report("sklearn.SGDRegressor (градиентный спуск) - TRAIN", rep_sgd_tr)
    print_report("sklearn.SGDRegressor (градиентный спуск) - TEST ", rep_sgd_te)

    # 7) (Референс) Закрытая формула (нормальные уравнения) - LinearRegression
    #   Куда должны прийти обе реализации
    ols = LinearRegression()
    ols.fit(X_tr_sc, y_tr)
    y_pred_tr_ols = ols.predict(X_tr_sc)
    y_pred_te_ols = ols.predict(X_te_sc)

    rep_ols_tr = regression_report(y_tr, y_pred_tr_ols)
    rep_ols_te = regression_report(y_te, y_pred_te_ols)

    print_report("sklearn.LinearRegression (референс) - TRAIN", rep_ols_tr)
    print_report("sklearn.LinearRegression (референс) - TEST ", rep_ols_te)

    # 8) Дополнительная интерпретация коэффициентов
    #    ВАЖНО: сравнивать коэффициенты корректно в одном и том же пространстве признаков.
    #           Ниже печатаем веса для scaled-признаков (сопоставимы между моделями).
    print("\n--- Коэффициенты (scaled space) ---")
    w_gd = np.asarray(model_gd.w_)
    b_gd = float(model_gd.b_)
    print(f"GD  : bias={b_gd:.6f}, weights={np.round(w_gd, 6)}")

    print(f"SGD : bias={sgd.intercept_[0]:.6f}, weights={np.round(sgd.coef_, 6)}")
    print(f"OLS : bias={ols.intercept_:.6f}, weights={np.round(ols.coef_, 6)}")

    # 9) Короткие выводы:
    #    - Если обучение прошло корректно, метрики GD ~ SGD ~ OLS на test будут очень близки.
    #    - Различия между GD и SGD связаны со стохастическим характером обновлений в SGD.
    #    - Масштабирование -> критично для стабильной сходимости GD/SGD.
    print("\n=== Итог ===")
    print("1) Реализован батчевый градиентный спуск (с нуля), нотация y_pred - везде.")
    print("2) Проведено сравнение с библиотечным градиентным спуском (SGDRegressor).")
    print("3) Для контроля показаны метрики закрытой формулы (LinearRegression).")
    print('4) Результаты на тесте сопоставимы -> "ручная" реализация градиентного спуска корректна.')


if __name__ == "__main__":
    main()
