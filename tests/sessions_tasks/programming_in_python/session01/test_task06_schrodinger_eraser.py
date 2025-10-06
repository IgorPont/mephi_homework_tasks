"""
Тесты к заданию 6.

Что проверяем:
    - Длина строки сохраняется.
    - Количество стертых слов соответствует формуле max(1, floor(N * percent / 100)).
    - Работа с файлами: читаем -> стираем -> записываем.
    - Пустой текст и текст без слов не ломают код.

Примечание:
    считаем стертые слова строго по позициям слов в исходном тексте,
    а не по количеству групп точек — чтобы не учитывать пунктуацию.
"""

import re
from pathlib import Path

from sessions_tasks.programming_in_python.session01.task06_schrodinger_eraser import (
    erase_random_words,
    erase_in_file,
)

# Регулярное выражение для поиска слов любых языков
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _word_spans(text: str) -> list[tuple[int, int]]:
    """
    Возвращает список (start, end) для всех слов в исходном тексте.
    """
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _count_words(text: str) -> int:
    """
    Считает количество слов в тексте.
    """
    return len(_word_spans(text))


def _count_erased_words_by_spans(src: str, redacted: str) -> int:
    """
    Считает, сколько именно исходных слов было стерто, то есть считаем последовательность точек,
    но без учета пунктуации.
    """
    erased = 0
    for start, end in _word_spans(src):
        if all(ch == "." for ch in redacted[start:end]):
            erased += 1
    return erased


def test_erase_random_words_basic():
    """
    Должно стереться процентное отношение слов, длина строки не меняется.
    """
    text = "Сегодня солнечный день, и я собираюсь гулять в парке с моими друзьями."
    # всего слов
    n_words = _count_words(text)
    percent = 30
    expected_erased = max(1, n_words * percent // 100)

    result = erase_random_words(text, percent)

    assert len(result) == len(text), "Длина результата должна совпадать с исходной"
    # считаем стертые слова (без учета пунктуации)
    assert _count_erased_words_by_spans(text, result) == expected_erased, "Стертое число слов не совпало с ожиданием"
    # проверяем замену
    assert "." in result


def test_erase_random_words_no_words_keeps_text():
    """
    Если только знаки препинания, возвращаем исходный текст.
    """
    text = ".,!? — ;: …"
    result = erase_random_words(text, percent=50)
    assert result == text


def test_erase_random_words_empty_string():
    """
    Пустая строка возвращается.
    """
    assert erase_random_words("", percent=50) == ""


def test_erase_in_file_io(tmp_path: Path):
    """
    Проверка работы с файлами: читаем -> стираем -> записываем.
    """
    input_text = "Тестовый файл со словами и парой лишних, но полезных знаков."
    input_file = tmp_path / "in.txt"
    output_file = tmp_path / "out.txt"
    input_file.write_text(input_text, encoding="utf-8")

    erase_in_file(input_file, output_file, percent=50)

    assert output_file.exists(), "Выходной файл должен быть создан"
    result = output_file.read_text(encoding="utf-8")
    assert len(result) == len(input_text), "Длина результата должна совпадать с исходной"
    assert "." in result
