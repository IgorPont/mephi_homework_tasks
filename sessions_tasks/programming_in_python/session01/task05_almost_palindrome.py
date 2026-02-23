"""
Задание 5. Почти палиндром.

Условие:
    Напишите программу, которая проверяет, является ли строка палиндромом или "почти палиндромом".
    "Почти палиндром" означает, что можно удалить одну букву, чтобы строка стала палиндромом.

Примеры:
    "abba" -> палиндром
    "abca" -> почти палиндром (удаляем 'c')
    "abcdef" -> не палиндром
"""


def is_palindrome(s: str) -> bool:
    """
    Проверяет, является ли строка палиндромом.

    Аргументы:
        s (str): исходная строка.

    Возвращает:
        bool: True, если строка читается одинаково слева направо и справа налево.
    """
    return s == s[::-1]


def is_almost_palindrome(s: str) -> tuple[bool, str]:
    """
    Проверяет, является ли строка палиндромом или почти палиндромом.

    Аргументы:
        s (str): исходная строка.

    Возвращает:
        tuple[bool, str]: кортеж (определение и подтверждение) с одним из результатов:
            - "палиндром"
            - "почти палиндром"
            - "не палиндром"
    """
    if is_palindrome(s):
        return True, "палиндром"

    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            # пробуем удалить символ слева или справа
            s1 = s[left + 1 : right + 1]
            s2 = s[left:right]
            if s1 == s1[::-1] or s2 == s2[::-1]:
                return True, "почти палиндром"
            else:
                return False, "не палиндром"
        left += 1
        right -= 1

    return False, "не палиндром"


if __name__ == "__main__":
    """
    Запуск через Poetry:
        poetry run python sessions_tasks/programming_in_python/session01/task05_almost_palindrome.py
    """

    test_strings = ["abba", "abca", "abcdef", "racecar", "radkar", "madmam"]

    print("🔍 Проверка строк на палиндромность:\n")
    for s in test_strings:
        result, kind = is_almost_palindrome(s)
        print(f"{s:<10} -> {kind}")
