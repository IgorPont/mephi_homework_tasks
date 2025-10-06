"""
Тесты для модуля task02_string_compress.py

Проверяются:
    - базовые случаи с повторяющимися символами;
    - случаи, когда сжатие не уменьшает длину;
    - пустая строка и спецсимволы;
    - корректное форматирование и длина результата.
"""

import pytest
from sessions_tasks.programming_in_python.session01.task02_string_compress import compress_string


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("aaabbc", "a3b2c"),
        ("abcd", "abcd"),
        ("", ""),
        ("    ", " 4"),
        ("!!**", "!!**"),
        ("aabccccaaa", "a2bc4a3"),
        ("111112222233", "152532"),
        ("xxxYYzzzZ", "x3Y2z3Z"),
    ],
)
def test_compress_string_basic(input_text: str, expected: str):
    """
    Проверка базовых случаев.
    """
    assert compress_string(input_text) == expected


def test_compress_string_empty():
    """
    Проверяет корректность обработки пустой строки.
    Функция должна вернуть пустую строку без ошибок.
    """
    assert compress_string("") == ""


def test_compress_string_not_shorter():
    """
    Проверяет, что функция возвращает исходную строку,
    если сжатая по длине не меньше исходной.
    """
    text = "abab"  # нет серий для сжатия
    result = compress_string(text)
    assert result == text
    assert len(result) == len(text)


def test_compress_string_special_symbols():
    """
    Проверяет корректную работу функции со спецсимволами.
    Проверяется формирование серий и корректность итогового результата.
    """
    text = "###$$$@@"
    expected = "#3$3@2"
    assert compress_string(text) == expected
