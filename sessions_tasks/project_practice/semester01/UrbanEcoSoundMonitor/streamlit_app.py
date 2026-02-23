"""
Streamlit-приложение UrbanEcoSoundMonitor.

Прототип мониторинга городской экосистемы по акустическим данным на основе
предобученной модели AST (Audio Spectrogram Transformer) из библиотеки transformers.

Основные идеи:
    - Пользователь выбирает папку с .wav-файлами (2–4 секунды каждый).
    - Для каждого файла считаем top-k предсказаний модели.
    - Строим простую статистику по классам звуков.
    - Подсвечиваем шумные или стрессовые классы (транспорт, сирены) как потенциальный риск.

Используемая модель:
    xpariz10/ast-finetuned-audioset-10-10-0.4593-finetuning-ESC-50

Примечание:
    AST-модель дообучена на датасете ESC-50 (environmental sound classification).
"""

import dataclasses
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from transformers import pipeline

# --- Константы и настройки ---

# Имя модели на Hugging Face
MODEL_NAME = "xpariz10/ast-finetuned-audioset-10-10-0.4593-finetuning-ESC-50"

# Поддерживаемые расширения аудио
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")

# Целевой sample rate для модели (AST обычно обучен на 16 kHz)
TARGET_SAMPLE_RATE = 16000

# Ключевые слова шумных/стрессовых классов (английские имена из модели)
NOISY_KEYWORDS = [
    "Siren",
    "Car",
    "Vehicle",
    "Traffic",
    "Engine",
    "Motorcycle",
    "Truck",
    "Subway",
    "Train",
    "Emergency vehicle",
    "Helicopter",
]

# Упрощенный словарь переводов классов ESC-50 на русский.
# Если какого-то класса нет в словаре, будет отображаться английское название.
LABEL_TRANSLATIONS: dict[str, str] = {
    "Whoop": "Возглас / вскрик",
    "Static": "Статический шум",
    "Laughter": "Смех",
    "Speech": "Речь",
    "Children shouting": "Крики детей",
    "Sine wave": "Синусоидальный сигнал",
    "Siren": "Сирена",
    "Car": "Автомобиль",
    "Vehicle": "Транспортное средство",
    "Train": "Поезд",
    "Helicopter": "Вертолет",
    "Dog": "Лай собаки",
    # Можно расширять по мере надобности
}

# Более подробные русские описания некоторых классов
LABEL_RU: dict[str, str] = {
    "Siren": "Сирена (машина скорой, полиция, пожарные)",
    "Car horn": "Автомобильный сигнал",
    "Engine": "Работающий двигатель",
    "Train": "Звук поезда",
    "Subway": "Метро / подземный поезд",
    "Helicopter": "Вертолет",
    "Children shouting": "Крики детей / шум детской площадки",
    "Whoop": "Возглас 'у-у-у' / эмоциональный крик",
    "Static": "Статический шум / помехи",
    "Sine wave": "Синусоидальный тон (технический сигнал)",
    "Speech": "Речь человека",
    "Laughter": "Смех",
    "Rain": "Шум дождя",
    "Wind": "Ветер",
    "Fireworks": "Фейерверки / салют",
    "Dog": "Лай собаки",
    "Dog bark": "Лай собаки",
    "Glass": "Звон/разбитие стекла",
}


# --- Вспомогательные структуры данных ---


@dataclasses.dataclass
class ClipPrediction:
    """
    Результат инференса для одного аудио-файла.

    Атрибуты:
        file_path: Path
            Путь к исходному аудио-файлу.
        top_labels: List[Dict[str, Any]]
            Список словарей от transformers.pipeline с полями:
            - label: название класса (EN)
            - score: вероятность (0..1)
        has_noisy_label: bool
            Флаг: есть ли среди top-k классов шумный или стрессовый класс.
    """

    file_path: Path
    top_labels: list[dict[str, Any]]
    has_noisy_label: bool


# --- Утилиты работы с метками ---


def translate_label(label: str) -> str:
    """
    Возвращает человеко-понятное русское описание класса.

    Сначала пытаемся найти в словаре LABEL_RU.
    Если не нашли — аккуратно преобразуем исходное имя:
        заменяем '_' на пробелы и оставляем английский текст.
    """
    if label in LABEL_RU:
        return LABEL_RU[label]

    # Простейший fallback: разбиваем snake_case
    return label.replace("_", " ")


# --- Утилиты работы с аудио ---


def load_audio(file_path: Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Загружает аудио-файл и при необходимости ресемплирует (преобразует) его.

    Используем librosa.load, который:
        - читает популярные форматы (wav, flac, ogg, mp3),
        - сразу ресемплирует до target_sample_rate,
        - сводит стерео к моно.

    Возвращает:
        waveform_np: np.ndarray
            Массив формы (num_samples,) в float32.
        sample_rate: int
            Фактический sample rate (совпадает с target_sample_rate).
    """
    waveform_np, sample_rate = librosa.load(
        path=str(file_path),
        sr=target_sample_rate,
        mono=True,
    )
    waveform_np = waveform_np.astype(np.float32)
    return waveform_np, sample_rate


# --- Модель и инференс ----


@st.cache_resource(show_spinner="Загружаю предобученную модель AST с Hugging Face...")
def load_audio_classifier(device: str = "cpu"):
    """
    Загружает и кэширует аудио-классификатор на базе AST.

    Параметры:
        device: str
            "cpu" или "cuda". Для локального запуска на ноутбуке обычно достаточно "cpu".

    Возвращает:
        audio_pipe: transformers.pipelines.Pipeline
            Пайплайн `audio-classification`, готовый к инференсу.
    """
    if device == "cuda":
        device_index = 0
    else:
        # -1 = CPU
        device_index = -1

    audio_pipe = pipeline(
        task="audio-classification",
        model=MODEL_NAME,
        device=device_index,
    )
    return audio_pipe


def classify_clip(file_path: Path, audio_pipe, top_k: int = 5) -> ClipPrediction:
    """
    Запускает инференс модели для одного аудио-клипа.

    Параметры:
        file_path: Path
            Путь к аудио-файлу.
        audio_pipe:
            Объект transformers.pipeline("audio-classification").
        top_k: int
            Сколько top-классов возвращать.

    Возвращает:
        ClipPrediction
            Структурированный результат инференса.
    """
    waveform, sample_rate = load_audio(file_path)

    # Формируем входные данные в формате, который понимает audio-pipeline
    model_input = {
        "array": waveform,
        "sampling_rate": sample_rate,
    }

    # Получаем top-k предсказаний
    raw_predictions = audio_pipe(model_input, top_k=top_k)

    # Проверяем, есть ли среди предсказаний шумные классы (по английским ключевым словам)
    has_noisy = any(
        any(keyword.lower() in pred["label"].lower() for keyword in NOISY_KEYWORDS) for pred in raw_predictions
    )

    return ClipPrediction(
        file_path=file_path,
        top_labels=raw_predictions,
        has_noisy_label=has_noisy,
    )


def analyze_folder(audio_dir: Path, audio_pipe, top_k: int = 5) -> list[ClipPrediction]:
    """
    Запускает инференс для всех файлов в указанной папке.

    Параметры:
        audio_dir: Path
            Папка с аудио-файлами (один файл = одна короткая сцена 2–4 секунды).
        audio_pipe:
            Объект transformers.pipeline("audio-classification").
        top_k: int
            Сколько top-классов возвращать для каждого файла.

    Возвращает:
        List[ClipPrediction]
            Список результатов по всем файлам.
    """
    predictions: list[ClipPrediction] = []

    for file_path in sorted(audio_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        prediction = classify_clip(file_path, audio_pipe, top_k=top_k)
        predictions.append(prediction)

    return predictions


# --- Визуализация статистики ---


def build_summary_dataframe(predictions: list[ClipPrediction]) -> pd.DataFrame:
    """
    Строит сводную таблицу по результатам инференса.

    Каждая строка -> один аудио-файл.

    Технические колонки:
        - file: имя файла
        - top_label_en: главный предсказанный класс (EN)
        - top_label: главный предсказанный класс (RU, если есть перевод)
        - top_score: вероятность top-класса
        - noisy_flag: 1, если среди top-k есть шумный класс

    Пользовательские (русские) колонки для удобства чтения в UI:
        - Файл
        - Класс (EN)
        - Класс (RUS)
        - Вероятность
        - Шумовой класс (0/1)
    """
    rows: list[dict[str, Any]] = []

    for item in predictions:
        top = item.top_labels[0] if item.top_labels else {"label": "N/A", "score": 0.0}
        top_label_en = top["label"]
        top_label_ru = LABEL_TRANSLATIONS.get(top_label_en, top_label_en)
        top_score = float(top["score"])
        noisy_flag = int(item.has_noisy_label)

        rows.append(
            {
                # технические поля
                "file": item.file_path.name,
                "top_label_en": top_label_en,
                "top_label": top_label_ru,
                "top_score": top_score,
                "noisy_flag": noisy_flag,
                # русские поля для UI
                "Файл": item.file_path.name,
                "Класс (EN)": top_label_en,
                "Класс (RUS)": top_label_ru,
                "Вероятность": top_score,
                "Шумовой класс (0/1)": noisy_flag,
            }
        )

    return pd.DataFrame(rows)


def plot_label_distribution(df: pd.DataFrame) -> None:
    """
    Рисует bar-chart распределения топ-классов по всем файлам.

    Для оси X используем английские названия классов (top_label_en),
    чтобы сохранить соответствие с оригинальной моделью.
    """
    if df.empty:
        st.info("Нет данных для построения графика")
        return

    # Если есть колонка с английскими названиями, то используем ее,
    # иначе возвращаемся к старому поведению (top_label).
    label_column = "top_label_en" if "top_label_en" in df.columns else "top_label"

    label_counts = df[label_column].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots()
    label_counts.plot(kind="bar", ax=ax)
    ax.set_title("Распределение топ-классов по аудио-файлам")
    ax.set_xlabel("Класс (EN)")
    ax.set_ylabel("Количество файлов")
    plt.xticks(rotation=90)
    plt.tight_layout()

    st.pyplot(fig)


def plot_noisy_share(df: pd.DataFrame) -> None:
    """
    Показывает долю файлов с шумными или стрессовыми звуками
    """
    if df.empty:
        return

    # Берем русскую колонку, которую мы добавили в build_summary_dataframe
    noisy_share = df["Шумовой класс (0/1)"].mean()

    st.metric(
        label="Доля файлов с потенциально шумными/стрессовыми звуками",
        value=f"{noisy_share * 100:.1f} %",
    )


# --- UI Streamlit ---


def sidebar_controls() -> tuple[str, Path | None]:
    """
    Рисует элементы управления в боковой панели и возвращает выбор пользователя.

    Возвращает:
        device_choice: str
            Выбранное устройство инференса: cpu или cuda.
        audio_dir: Path | None
            Путь к папке с аудио-файлами (если пользователь указал корректный путь).
    """
    st.sidebar.header("⚙️ Настройки")

    device_choice = st.sidebar.selectbox(
        "Устройство инференса",
        options=["cpu", "cuda"],
        index=0,
        help="Для локального запуска на ноутбуке обычно достаточно CPU",
    )

    # По умолчанию -> стандартная папка проекта с демонстрационными файлами
    default_dir = Path.cwd() / "data" / "audio_samples"
    audio_dir_str = st.sidebar.text_input(
        "Папка с аудио (.wav/.flac/.mp3/.ogg)",
        value=str(default_dir),
        help="Укажи папку, где лежат короткие аудио-клипы (2–4 секунды каждый)",
    )

    audio_dir = Path(audio_dir_str).expanduser()
    if not audio_dir.exists():
        st.sidebar.warning("Указанная папка пока не существует")
        return device_choice, None

    st.sidebar.info(
        "Совет: положи свои .wav-файлы (2–4 секунды) в указанную папку, " "они появятся в списке для пакетного анализа",
    )

    return device_choice, audio_dir


def main() -> None:
    """
    Точка входа Streamlit-приложения.
    """
    st.set_page_config(
        page_title="UrbanEcoSoundMonitor",
        page_icon="🌿",
        layout="wide",
    )

    device_choice, audio_dir = sidebar_controls()

    # Заголовок и описание
    st.title("🌿 UrbanEcoSoundMonitor")
    st.subheader("Прототип мониторинга городской экосистемы по акустическим данным")

    st.markdown(
        """
        **Что делает этот прототип:**

        - Использует предобученную модель `AST` из библиотеки `transformers`
          (`xpariz10/ast-finetuned-audioset-10-10-0.4593-finetuning-ESC-50`).
        - Классифицирует короткие аудио-файлы на разнообразные классы окружающих звуков
          (транспорт, природа, человеческая активность, технические сигналы).
        - Строит простую статистику по классам и подсвечивает долю шумных или стрессовых звуков,
          важных для мониторинга качества городской экосистемы.
        """
    )

    st.info(
        "Для демонстрации необходимо разместить несколько коротких .wav-файлов "
        "в папке `data/audio_samples` и нажать кнопку анализа.",
    )

    if audio_dir is None:
        st.stop()

    # Загружаем модель (один раз за сессию) с выбранным устройством
    audio_pipe = load_audio_classifier(device=device_choice)

    # Кнопка запуска анализа
    run_analysis = st.button("🚀 Запустить анализ папки")

    if run_analysis:
        audio_files = [f for f in sorted(audio_dir.iterdir()) if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]

        if not audio_files:
            st.warning("В папке нет подходящих аудио-файлов")
            st.stop()

        st.write(f"Найдено файлов для анализа: **{len(audio_files)}**")

        predictions: list[ClipPrediction] = []
        progress_bar = st.progress(0.0)

        for idx, file_path in enumerate(audio_files, start=1):
            with st.spinner(f"Анализирую {file_path.name}..."):
                prediction = classify_clip(file_path, audio_pipe, top_k=5)
                predictions.append(prediction)

            progress_bar.progress(idx / len(audio_files))

        st.success("✅ Анализ завершен")

        # Строим таблицу и кладем все в session_state
        df_summary = build_summary_dataframe(predictions)
        st.session_state["df_summary"] = df_summary
        st.session_state["predictions"] = predictions

    # --- Блок отображения результатов (живет вне кнопки) ---

    if st.session_state.get("df_summary") is not None:
        df_summary: pd.DataFrame = st.session_state["df_summary"]
        predictions: list[ClipPrediction] = st.session_state["predictions"]

        st.subheader("📊 Сводная таблица по аудио-файлам")
        st.dataframe(df_summary, use_container_width=True)

        st.subheader("📈 Распределение топ-классов")
        plot_label_distribution(df_summary)

        st.subheader("🌡️ Индикатор шумовой нагрузки")
        plot_noisy_share(df_summary)

        # Детальный просмотр одного файла
        st.subheader("🔍 Детальный просмотр предсказаний по одному файлу")
        selected_file = st.selectbox(
            "Выбери файл для детального просмотра:",
            options=df_summary["Файл"].tolist(),
        )

        selected = next(item for item in predictions if item.file_path.name == selected_file)

        st.write(f"**Файл:** `{selected.file_path.name}`")
        st.write("Top-5 предсказаний модели:")

        detail_rows = [
            {
                "Класс (EN)": p["label"],
                "Класс (RUS)": translate_label(p["label"]),
                "Вероятность": float(p["score"]),
            }
            for p in selected.top_labels
        ]
        st.table(pd.DataFrame(detail_rows))

        if selected.has_noisy_label:
            st.warning(
                "В числе top-классов присутствуют шумные или стрессовые звуки "
                "(транспорт, сирены), необходимо обратить внимание на данную локацию"
            )
        else:
            st.success(
                "В топ-классах не выявлено выраженных шумовых или стрессовых звуков, "
                "акустическая обстановка выглядит спокойной"
            )
    else:
        st.info('Нажми кнопку **"🚀 Запустить анализ папки"**, чтобы запустить инференс')


if __name__ == "__main__":
    main()
