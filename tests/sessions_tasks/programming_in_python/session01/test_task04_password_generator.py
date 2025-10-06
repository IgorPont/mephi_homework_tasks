"""
Тесты для модуля task04_password_generator.py

Проверяются:
    - корректность длины пароля;
    - наличие всех типов символов (буквы, цифры, спецсимволы);
    - уникальность при повторных генерациях;
    - корректная обработка недопустимой длины.
"""

import pytest
from sessions_tasks.programming_in_python.session01.task04_password_generator import generate_password


def test_generate_password_length():
    """
    Проверяет, что длина сгенерированного пароля соответствует указанной.
    """
    for length in [4, 8, 12, 20]:
        pwd = generate_password(length)
        assert len(pwd) == length


def test_generate_password_content():
    """
    Проверяет, что в пароле присутствует хотя бы:
        - одна заглавная буква,
        - одна строчная буква,
        - одна цифра,
        - один специальный символ.
    """
    pwd = generate_password(12)
    assert any(c.islower() for c in pwd)
    assert any(c.isupper() for c in pwd)
    assert any(c.isdigit() for c in pwd)
    assert any(c in "!@#$%^&*()_+-=[]{};:'\",.<>?/\\|" for c in pwd)


def test_generate_password_invalid_length():
    """
    Проверяет, что при длине меньше 4 выбрасывается исключение ValueError.
    """
    with pytest.raises(ValueError):
        generate_password(3)


def test_generate_password_randomness():
    """
    Проверяет, что при повторных вызовах функция генерирует разные пароли.
    """
    pwd1 = generate_password(10)
    pwd2 = generate_password(10)
    assert pwd1 != pwd2
