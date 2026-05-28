"""
Подготовка признаков и целевых переменных для задач курсовой работы.

Модуль содержит функции, которые формируют матрицу признаков "X"
и целевые переменные "y" для задач регрессии и классификации.

Пример использования:
    poetry run python -c "from sessions_tasks.classical_ml.session02.coursework_classical_ml.src.data_loader \
    import load_coursework_dataset; \
    from sessions_tasks.classical_ml.session02.coursework_classical_ml.src.preprocessing \
    import build_features, build_regression_target, build_si_above_threshold_target; \
    from sessions_tasks.classical_ml.session02.coursework_classical_ml.src.config import TARGET_IC50; \
    df = load_coursework_dataset(); X = build_features(df); y_reg = build_regression_target(df, TARGET_IC50); \
    y_cls = build_si_above_threshold_target(df); print('X:', X.shape); print('y_reg:', y_reg.shape); \
    print('y_cls:', y_cls.value_counts().to_dict()); print(X.columns[:10].tolist())"

Пример вывода:
        X: (1001, 210)
        y_reg: (1001,)
        y_cls: {0: 644, 1: 357}
        [
            'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex',
            'MinEStateIndex', 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt',
            'ExactMolWt', 'NumValenceElectrons'
        ]

    Расшифровка:
        - X: (1001, 210) -> Датасет содержит 1001 объект и 210 признаков

        - y_reg: (1001,) -> Для каждого объекта существует одно числовое значение целевой переменной
            для задачи регрессии.

        - y_cls: {0: 644, 1: 357} -> Распределение классов целевой переменной такое:
            644 объекта относятся к классу 0,
            357 объектов относятся к классу 1.

        - Список признаков -> Показывает первые признаки из матрицы X для проверки,
            что признаки были корректно сформированы.

Пример запуска обработки константных и сильно коррелирующих признаков:
    poetry run python -c "from sessions_tasks.classical_ml.session02.coursework_classical_ml.src.data_loader \
    import load_coursework_dataset; \
    from sessions_tasks.classical_ml.session02.coursework_classical_ml.src.preprocessing \
    import build_features, preprocess_features; df = load_coursework_dataset(); X = build_features(df); \
    X_clean, report = preprocess_features(X); print('До:', X.shape); \
    print('После:', X_clean.shape); print('Удалено константных:', len(report['constant_features'])); \
    print('Удалено почти константных:', len(report['near_constant_features'])); \
    print('Удалено коррелирующих:', len(report['highly_correlated_features'])); \
    print('Всего удалено:', len(report['removed_features']))"

Пример вывода:
    До: (1001, 210)
    После: (1001, 134)
    Удалено константных: 18
    Удалено почти константных: 15
    Удалено коррелирующих: 43
    Всего удалено: 76
"""

import pandas as pd

from sessions_tasks.classical_ml.session02.coursework_classical_ml.src.config import (
    TARGET_CC50,
    TARGET_IC50,
    TARGET_SI,
)


def get_service_columns(dataframe: pd.DataFrame) -> list[str]:
    """
    Определяет служебные столбцы, которые не являются признаками.
    В Excel-файле курсовой обнаружен безымянный первый столбец,
    который является сохраненным индексом DataFrame.
    При чтении через pandas такой столбец получает имя "Unnamed: 0".

    Аргументы:
        - dataframe: Исходный DataFrame.

    Возвращает:
        - Список служебных столбцов.
    """

    service_columns = []

    for column in dataframe.columns:
        if str(column).startswith("Unnamed:"):
            service_columns.append(column)

    return service_columns


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    """
    Получает список признаков, доступных для обучения моделей.

    Из признакового пространства исключаются:
        - служебные индексные столбцы вида "Unnamed:*"
        - целевые переменные IC50, CC50 и SI

    Аргументы:
        - dataframe: Исходный DataFrame.

    Возвращает:
        - Список названий столбцов, которые используются в качестве признаков.
    """

    service_columns = get_service_columns(dataframe)
    excluded_columns = set(service_columns + [TARGET_IC50, TARGET_CC50, TARGET_SI])

    return [column for column in dataframe.columns if column not in excluded_columns]


def build_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует матрицу признаков "X".

    Аргументы:
        - dataframe: Исходный DataFrame.

    Возвращает:
        - DataFrame только с признаками.
    """

    feature_columns = get_feature_columns(dataframe)
    return dataframe[feature_columns].copy()


def build_regression_target(dataframe: pd.DataFrame, target_column: str) -> pd.Series:
    """
    Формирует целевую переменную для задачи регрессии.

    Аргументы:
        - dataframe: Исходный DataFrame.
        - target_column: Название целевого столбца.

    Возвращает:
        - Series с целевой переменной.

    Исключения:
        - ValueError: Если целевой столбец отсутствует в DataFrame.
    """

    if target_column not in dataframe.columns:
        raise ValueError(f"Целевой столбец отсутствует в DataFrame: {target_column}")

    return dataframe[target_column].copy()


def build_ic50_above_median_target(dataframe: pd.DataFrame) -> pd.Series:
    """
    Формирует целевой класс -> IC50 выше медианного значения.
    Series -> один столбец данных из DataFrame,
    который используется как то, что модель должна предсказывать.

    Аргументы:
        - dataframe: Исходный DataFrame.

    Возвращает:
        - Series с бинарной целевой переменной:
            1 -> значение IC50 выше медианы
            0 -> значение IC50 ниже либо равно медиане
    """

    median_value = dataframe[TARGET_IC50].median()
    return (dataframe[TARGET_IC50] > median_value).astype(int)


def build_cc50_above_median_target(dataframe: pd.DataFrame) -> pd.Series:
    """
    Формирует целевой класс -> CC50 выше медианного значения.
    Series -> один столбец данных из DataFrame,
    который используется как то, что модель должна предсказывать.

    Аргументы:
        - dataframe: Исходный DataFrame.

    Возвращает:
        - Series с бинарной целевой переменной:
            1 -> значение CC50 выше медианы
            0 -> значение CC50 ниже либо равно медиане
    """

    median_value = dataframe[TARGET_CC50].median()
    return (dataframe[TARGET_CC50] > median_value).astype(int)


def build_si_above_median_target(dataframe: pd.DataFrame) -> pd.Series:
    """
    Формирует целевой класс -> SI выше медианного значения.
    Series -> один столбец данных из DataFrame,
    который используется как то, что модель должна предсказывать.

    Аргументы:
        - dataframe: Исходный DataFrame.

    Возвращает:
        - Series с бинарной целевой переменной:
            1 -> значение SI выше медианы
            0 -> значение SI ниже либо равно медиане
    """

    median_value = dataframe[TARGET_SI].median()
    return (dataframe[TARGET_SI] > median_value).astype(int)


def build_si_above_threshold_target(dataframe: pd.DataFrame, threshold: float = 8.0) -> pd.Series:
    """
    Формирует целевой класс -> SI выше заданного порога.
    Series -> один столбец данных из DataFrame,
    который используется как то, что модель должна предсказывать.

    Аргументы:
        - dataframe: Исходный DataFrame.
        - threshold: Пороговое значение SI.

    Возвращает:
        - Series с бинарной целевой переменной:
            1 -> значение SI выше порога
            0 -> значение SI ниже либо равно порогу
    """

    return (dataframe[TARGET_SI] > threshold).astype(int)


def get_constant_features(features: pd.DataFrame) -> list[str]:
    """
    Находит константные признаки.

    Константными признаками считаем признаки, у которых во всех строках одно и то же значение.
    Такой признак мешает модели различать объекты между собой и поэтому будет удален.

    Аргументы:
        - features: Матрица признаков X.

    Возвращает:
        - Список названий константных признаков.
    """

    return [column for column in features.columns if features[column].nunique(dropna=False) <= 1]


def get_near_constant_features(features: pd.DataFrame, threshold: float = 0.99) -> list[str]:
    """
    Находит почти константные признаки.

    Почти константным признаком считаем признак, у которого одно значение встречается
    как минимум в threshold-доле строк.
    Например, при threshold=0.99 одно и то же
    значение должно встречаться минимум в 99% наблюдений.

    Такие признаки содержат очень мало полезной информации для модели.

    Аргументы:
        - features: Матрица признаков X.
        - threshold: Минимальная доля самого частого значения.

    Возвращает:
        - Список названий почти константных признаков.
    """

    near_constant_features = []

    for column in features.columns:
        dominant_value_share = features[column].value_counts(dropna=False, normalize=True).max()

        if dominant_value_share >= threshold and features[column].nunique(dropna=False) > 1:
            near_constant_features.append(column)

    return near_constant_features


def get_highly_correlated_features(
    features: pd.DataFrame,
    threshold: float = 0.95,
) -> list[str]:
    """
    Определяет признаки с высокой взаимной корреляцией.

    Если два признака сильно коррелируют между собой, один из них можно удалить,
    так как он несет почти ту же информацию, что и другой признак.

    Для поиска используется Spearman-корреляция, так как она устойчивее к выбросам
    и оценивает монотонную связь между признаками.

    Логика отбора:
        - строим корреляционную матрицу по признакам
        - рассматриваем только верхний треугольник матрицы, чтобы не проверять пары дважды
        - если абсолютная корреляция пары признаков >= threshold, то
            добавляем второй признак пары в список на удаление.

    Аргументы:
        - features: DataFrame с признаками.
        - threshold: Порог абсолютной корреляции для удаления признака.

    Возвращает:
        - Список признаков, которые рекомендуется удалить из-за высокой корреляции.
    """

    if features.empty:
        return []

    correlation_matrix = features.corr(method="spearman").abs()

    highly_correlated_features = set()

    columns = correlation_matrix.columns.tolist()

    for current_column_index in range(len(columns)):
        current_column = columns[current_column_index]

        for previous_column_index in range(current_column_index):
            previous_column = columns[previous_column_index]
            correlation_value = correlation_matrix.loc[previous_column, current_column]

            if pd.notna(correlation_value) and correlation_value >= threshold:
                highly_correlated_features.add(current_column)
                break

    return sorted(highly_correlated_features)


def preprocess_features(
    features: pd.DataFrame,
    near_constant_threshold: float = 0.99,
    correlation_threshold: float = 0.9,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Выполняет предобработку признаков по итогам EDA.

    Из матрицы признаков удаляются:
        - константные признаки
        - почти константные признаки
        - признаки, сильно коррелирующие с другими признаками

    Примечание:
        Предобработка применяется только к признакам X.
        Целевые переменные в эту функцию не передаем.

    Аргументы:
        - features: Исходная матрица признаков X.
        - near_constant_threshold: Порог для поиска почти константных признаков.
        - correlation_threshold: Порог для поиска сильно коррелирующих признаков.

    Возвращает:
        - Очищенная матрица признаков.
        - Словарь с информацией об удаленных признаках.
    """

    cleaned_features = features.copy()

    constant_features = get_constant_features(cleaned_features)
    cleaned_features = cleaned_features.drop(columns=constant_features)

    near_constant_features = get_near_constant_features(
        cleaned_features,
        threshold=near_constant_threshold,
    )
    cleaned_features = cleaned_features.drop(columns=near_constant_features)

    highly_correlated_features = get_highly_correlated_features(
        cleaned_features,
        threshold=correlation_threshold,
    )
    cleaned_features = cleaned_features.drop(columns=highly_correlated_features)

    preprocessing_report = {
        "constant_features": constant_features,
        "near_constant_features": near_constant_features,
        "highly_correlated_features": highly_correlated_features,
        "removed_features": constant_features + near_constant_features + highly_correlated_features,
        "remaining_features": cleaned_features.columns.tolist(),
    }

    return cleaned_features, preprocessing_report
