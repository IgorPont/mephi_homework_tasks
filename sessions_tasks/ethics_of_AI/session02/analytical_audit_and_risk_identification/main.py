"""
Итоговый проект по дисциплине "Этика искусственного интеллекта".

Тема:
    Аудит и этическое сопровождение внедрения ИИ-системы для найма персонала
    "КадроБот 3000".

Назначение файла:
    Скрипт выполняет базовый технический аудит синтетических данных hiring_data.csv:
        - проверяет структуру датасета
        - анализирует решения модели по возрастным группам
        - выявляет признаки дискриминации кандидатов 45+
        - показывает возможные proxy-переменные возраста
        - рассчитывает простые метрики справедливости

Результаты этого анализа используются далее в аналитической записке
и презентации по проекту.

Пример запуска:
    poetry run python sessions_tasks/ethics_of_AI/session02/analytical_audit_and_risk_identification/main.py

Пример вывода:

================================================================================
1. Общая информация о датасете
================================================================================
Количество строк: 5000
Количество столбцов: 14

Список столбцов:
- id
- age
- age_group
- gender
- education
- experience
- graduation_year
- outdated_vocab
- hobby
- youth_hobby
- model_score
- model_decision
- true_success
- historical_success

Первые 5 строк:
   id  age age_group  gender  education  experience  graduation_year  outdated_vocab        hobby  youth_hobby  model_score  model_decision  true_success  historical_success
0   1   56       45+       1          3          24           1988.0               1  садоводство            0     0.227043               0             1                   0
1   2   46       45+       0          2          21           2001.0               0  путешествия            0     0.388544               0             1                   0
2   3   32     30-45       0          1           9           2016.0               1   фотография            0     0.353160               0             1                   1
3   4   60       45+       1          2          25           1982.0               1      вязание            0     0.140580               0             0                   0
4   5   25       <30       1          3           9           2022.0               1    видеоигры            1     0.788198               1             0                   1

================================================================================
2. Анализ по возрастным группам
================================================================================
  age_group  count  avg_age  avg_model_score  invite_rate  true_success_rate  historical_success_rate  outdated_vocab_rate  youth_hobby_rate  avg_graduation_year  avg_experience
0     30-45   1576    38.17            0.481        0.468              0.374                    0.492                0.348             0.000               2006.2           11.97
1       45+   2093    55.44            0.247        0.038              0.359                    0.213                0.729             0.000               1990.6           22.63
2       <30   1331    23.95            0.705        0.941              0.414                    0.741                0.139             0.445               2016.0            3.68

================================================================================
3. Метрики справедливости
================================================================================

Демографический паритет:
  age_group  invite_rate  demographic_parity_ratio_vs_best
0     30-45        0.468                             0.497
1       45+        0.038                             0.041
2       <30        0.941                             1.000

Равные возможности:
  age_group  true_positive_rate
0     30-45               0.508
1       45+               0.057
2       <30               0.947

================================================================================
4. Анализ proxy-переменных
================================================================================

Корреляция признаков с возрастом:
age                   1.000
experience            0.913
outdated_vocab        0.526
true_success         -0.037
historical_success   -0.420
youth_hobby          -0.485
model_decision       -0.764
model_score          -0.863
graduation_year      -0.884
Name: age, dtype: float64

Корреляция признаков с оценкой модели model_score:
model_score           1.000
graduation_year       0.878
model_decision        0.870
youth_hobby           0.576
historical_success    0.452
true_success          0.084
outdated_vocab       -0.699
experience           -0.805
age                  -0.863
Name: model_score, dtype: float64

================================================================================
5. Краткие выводы
================================================================================
1. Модель систематически занижает рейтинг кандидатов 45+, что видно по низкому среднему model_score и низкой доле приглашений.
2. Реальная успешность кандидатов 45+ по true_success не является нулевой и сопоставима с другими группами, поэтому массовое отклонение этой группы нельзя объяснить только низким качеством кандидатов.
3. Признаки graduation_year, experience и outdated_vocab могут работать как proxy-переменные возраста.
4. Исторические решения компании historical_success отражают прошлую практику найма и могут содержать уже существующую возрастную предвзятость.
5. Использовать модель как окончательный автоматический фильтр нельзя. На этапе внедрения необходимы человек в контуре, право кандидата на объяснение, процедура апелляции и регулярный аудит справедливости.
"""

from pathlib import Path

import pandas as pd

# Путь к текущей директории с заданием
BASE_DIR = Path(__file__).resolve().parent

# Путь к файлу с синтетическими данными
DATA_PATH = BASE_DIR / "hiring_data.csv"


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Загружает датасет с кандидатами.

    Аргументы:
        - file_path: путь к CSV-файлу

    Возвращает:
        - DataFrame с данными для анализа

    Исключения:
        - FileNotFoundError: если файл с данными не найден
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Файл {file_path.name} не найден. "
            f"Положите hiring_data.csv в директорию задания."
        )

    return pd.read_csv(file_path)


def print_dataset_info(df: pd.DataFrame) -> None:
    """
    Выводит общую информацию о датасете.

    Аргументы:
        - df: DataFrame с данными
    """

    print("=" * 80)
    print("1. Общая информация о датасете")
    print("=" * 80)

    print(f"Количество строк: {df.shape[0]}")
    print(f"Количество столбцов: {df.shape[1]}")

    print("\nСписок столбцов:")
    for column in df.columns:
        print(f"- {column}")

    print("\nПервые 5 строк:")
    print(df.head())


def analyze_by_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Анализирует ключевые показатели по возрастным группам.

    Аргументы:
        - df: DataFrame с данными.

    Возвращает таблицу с агрегированными метриками по возрастным группам.
    """

    age_group_metrics = (
        df.groupby("age_group")
        .agg(
            count=("id", "count"),
            avg_age=("age", "mean"),
            avg_model_score=("model_score", "mean"),
            invite_rate=("model_decision", "mean"),
            true_success_rate=("true_success", "mean"),
            historical_success_rate=("historical_success", "mean"),
            outdated_vocab_rate=("outdated_vocab", "mean"),
            youth_hobby_rate=("youth_hobby", "mean"),
            avg_graduation_year=("graduation_year", "mean"),
            avg_experience=("experience", "mean"),
        )
        .reset_index()
    )

    print("\n" + "=" * 80)
    print("2. Анализ по возрастным группам")
    print("=" * 80)

    print(
        age_group_metrics.round(
            {
                "avg_age": 2,
                "avg_model_score": 3,
                "invite_rate": 3,
                "true_success_rate": 3,
                "historical_success_rate": 3,
                "outdated_vocab_rate": 3,
                "youth_hobby_rate": 3,
                "avg_graduation_year": 1,
                "avg_experience": 2,
            }
        )
    )

    return age_group_metrics


def analyze_fairness_metrics(df: pd.DataFrame) -> None:
    """
    Рассчитывает базовые метрики справедливости.

    Используются:
        - demographic parity: сравнение доли приглашений между группами
        - equal opportunity: сравнение доли приглашенных среди реально успешных кандидатов

    Аргументы:
        - df: DataFrame с данными
    """

    print("\n" + "=" * 80)
    print("3. Метрики справедливости")
    print("=" * 80)

    invite_rates = (
        df.groupby("age_group")["model_decision"]
        .mean()
        .reset_index(name="invite_rate")
    )

    best_invite_rate = invite_rates["invite_rate"].max()
    invite_rates["demographic_parity_ratio_vs_best"] = (
        invite_rates["invite_rate"] / best_invite_rate
    )

    print("\nДемографический паритет:")
    print(invite_rates.round(3))

    successful_candidates = df[df["true_success"] == 1]

    equal_opportunity = (
        successful_candidates.groupby("age_group")["model_decision"]
        .mean()
        .reset_index(name="true_positive_rate")
    )

    print("\nРавные возможности:")
    print(equal_opportunity.round(3))


def analyze_proxy_variables(df: pd.DataFrame) -> None:
    """
    Анализирует признаки, которые могут выступать proxy-переменными возраста.

    Proxy-переменная -> признак, который формально не является запрещенным,
    но сильно связан с защищенным признаком, например с возрастом.

    Аргументы:
        - df: DataFrame с данными
    """

    print("\n" + "=" * 80)
    print("4. Анализ proxy-переменных")
    print("=" * 80)

    selected_columns = [
        "age",
        "graduation_year",
        "experience",
        "outdated_vocab",
        "youth_hobby",
        "model_score",
        "model_decision",
        "true_success",
        "historical_success",
    ]

    correlation_matrix = df[selected_columns].corr(numeric_only=True)

    print("\nКорреляция признаков с возрастом:")
    print(correlation_matrix["age"].sort_values(ascending=False).round(3))

    print("\nКорреляция признаков с оценкой модели model_score:")
    print(correlation_matrix["model_score"].sort_values(ascending=False).round(3))


def print_main_conclusions() -> None:
    """
    Выводит краткие выводы для переноса в аналитическую записку.
    """

    print("\n" + "=" * 80)
    print("5. Краткие выводы")
    print("=" * 80)

    conclusions = [
        (
            "Модель систематически занижает рейтинг кандидатов 45+, "
            "что видно по низкому среднему model_score и низкой доле приглашений."
        ),
        (
            "Реальная успешность кандидатов 45+ по true_success не является нулевой "
            "и сопоставима с другими группами, поэтому массовое отклонение этой группы "
            "нельзя объяснить только низким качеством кандидатов."
        ),
        (
            "Признаки graduation_year, experience и outdated_vocab могут работать "
            "как proxy-переменные возраста."
        ),
        (
            "Исторические решения компании historical_success отражают прошлую практику найма "
            "и могут содержать уже существующую возрастную предвзятость."
        ),
        (
            "Использовать модель как окончательный автоматический фильтр нельзя. "
            "На этапе внедрения необходимы человек в контуре, право кандидата на объяснение, "
            "процедура апелляции и регулярный аудит справедливости."
        ),
    ]

    for index, conclusion in enumerate(conclusions, start=1):
        print(f"{index}. {conclusion}")


def main() -> None:
    """
    Главная функция запуска анализа
    """

    df = load_data(DATA_PATH)

    print_dataset_info(df)
    analyze_by_age_group(df)
    analyze_fairness_metrics(df)
    analyze_proxy_variables(df)
    print_main_conclusions()


if __name__ == "__main__":
    main()
