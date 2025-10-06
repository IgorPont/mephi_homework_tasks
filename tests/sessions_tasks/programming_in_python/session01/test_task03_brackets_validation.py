"""
Тесты для модуля task03_brackets_validation.py

Проверяются:
    - корректные и некорректные комбинации скобок;
    - вложенные конструкции;
    - посторонние символы (буквы, цифры, операторы);
    - пустая строка.
"""

import pytest
from sessions_tasks.programming_in_python.session01.task03_brackets_validation import is_brackets_balanced


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("()", True),
        ("()[]{}", True),
        ("([{}])", True),
        ("([]{})", True),
        ("(]", False),
        ("([)]", False),
        ("(((", False),
        ("", True),
        ("abc(def)", True),
        ("{[()]}", True),
        ("a + (b * [c - {d / e}])", True),
        ("(a+b]*{c/d}", False),
    ],
)
def test_is_brackets_balanced_various_cases(input_text: str, expected: bool):
    """
    Проверяет корректность работы функции для различных комбинаций скобок.
    """
    assert is_brackets_balanced(input_text) == expected


def test_is_brackets_balanced_only_text():
    """
    Проверяет, что строка без скобок считается корректной.
    """
    assert is_brackets_balanced("hello world") is True


def test_is_brackets_balanced_unbalanced_mixed():
    """
    Проверяет случай несоответствия типов скобок.
    Пример: открывающая '(' и закрывающая ']'.
    """
    assert is_brackets_balanced("(]") is False
