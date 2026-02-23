"""
Тесты для модуля task05_almost_palindrome.py

Проверяются:
    - корректная идентификация палиндромов;
    - "почти палиндромов" (удалить один символ);
    - непалиндром (после удаления одного символа);
    - пустая строка, одиночный символ, длина 2;
    - работа с юникодом (кириллица, эмодзи);
    - соответствие формата результата: (bool, str) с метками "палиндром" / "почти палиндром" / "не палиндром".
"""

import pytest

from sessions_tasks.programming_in_python.session01.task05_almost_palindrome import (
    is_almost_palindrome,
    is_palindrome,
)


@pytest.mark.parametrize(
    "s, expected",
    [
        ("", True),
        ("a", True),
        ("aa", True),
        ("aba", True),
        ("abba", True),
        ("abc", False),
        ("ab", False),
        ("👍👍", True),
        ("казак", True),
        ("Казак", False),
    ],
)
def test_is_palindrome_basic(s: str, expected: bool):
    """
    Базовые проверки чистой палиндромности без удаления символов.
    """
    assert is_palindrome(s) is expected


@pytest.mark.parametrize(
    "s, expected_bool, expected_kind",
    [
        # Палиндромы
        ("", True, "палиндром"),
        ("a", True, "палиндром"),
        ("aba", True, "палиндром"),
        ("abba", True, "палиндром"),
        ("racecar", True, "палиндром"),
        # Почти палиндромы (удалить 1 символ)
        ("abca", True, "почти палиндром"),
        ("radkar", True, "почти палиндром"),
        ("madmam", True, "почти палиндром"),
        ("аааб", True, "почти палиндром"),
        ("abxyba", True, "почти палиндром"),
        # Не палиндромы
        ("abcdef", False, "не палиндром"),
        ("👍a👍b👍", False, "не палиндром"),
    ],
)
def test_is_almost_palindrome_various(s: str, expected_bool: bool, expected_kind: str):
    """
    Проверяет корректную классификацию строк
    """
    ok, kind = is_almost_palindrome(s)
    assert ok is expected_bool
    assert kind == expected_kind


def test_is_almost_palindrome_length_two():
    """
    Специальный случай длины 2:
        "aa" -> палиндром
        "ab" -> почти палиндром (удаляем 'a' или 'b' и получаем одиночный палиндром)
    """
    assert is_almost_palindrome("aa") == (True, "палиндром")
    assert is_almost_palindrome("ab") == (True, "почти палиндром")


def test_is_almost_palindrome_returns_tuple_format():
    """
    Проверяет формат возвращаемого значения: (bool, str) и допустимые метки.
    """
    result = is_almost_palindrome("xyz")
    assert isinstance(result, tuple) and len(result) == 2
    ok, kind = result
    assert isinstance(ok, bool)
    assert kind in {"палиндром", "почти палиндром", "не палиндром"}


def test_is_almost_palindrome_unicode_strict():
    """
    Проверяет, что функция не нормализует регистр/язык сама по себе:
    'Казак' (с заглавной К) не является палиндромом, но почти палиндромом не станет удалением одной буквы.
    """
    assert is_almost_palindrome("Казак") == (False, "не палиндром")
