"""
Задание 4. Генератор паролей.

Условие:
    Напишите функцию, которая генерирует пароль заданной длины.
    Пароль должен содержать буквы, цифры и специальные символы.
    Длина пароля задается пользователем.

Примеры:
    длина = 8 -> aG#2z@K7
    длина = 12 -> Q2@aZ7k!9b#L
"""

import random
import string


def generate_password(length_pass: int = 6) -> str:
    """
    Генерирует пароль заданной длины, включающий буквы, цифры и специальные символы.

    Аргументы:
        length_pass (int): длина генерируемого пароля (рекомендуется по умолчанию не менее 6 символов).

    Возвращает:
        str: случайно сгенерированный пароль.

    Алгоритм:
        1. Проверяем, что длина ≥ 4 (иначе невозможно включить все типы символов).
        2. Добавляем по одному символу из каждой категории.
        3. Дополняем пароль случайными символами из общего пула.
        4. Перемешиваем результат для случайного распределения.

    Пример ручного вызова:
        from sessions_tasks.programming_in_python.session01.task04_password_generator import generate_password
        print(generate_password(8)) # aG#2z@K7
        print(generate_password(12)) # Q2@aZ7k!9b#L
    """
    if length_pass < 4:
        raise ValueError("Длина пароля должна быть не менее 4 символов")

    # Наборы символов
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    symbols = string.punctuation

    # Гарантируем наличие хотя бы одного символа каждого типа
    password_chars = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(digits),
        random.choice(symbols),
    ]

    # Добавляем оставшиеся случайные символы
    all_chars = lowercase + uppercase + digits + symbols
    password_chars += random.choices(all_chars, k=length_pass - 4)

    # Перемешиваем для случайного порядка
    random.shuffle(password_chars)

    # Возвращаем итоговую строку
    return "".join(password_chars)


if __name__ == "__main__":
    """
    Запуск через poetry:
        poetry run python sessions_tasks/programming_in_python/session01/task04_password_generator.py

    """
    print("🧩 Результат генерации паролей:\n")
    for length_pass in [6, 8, 12, 16]:
        print(f"Длина {length_pass:2}: {generate_password(length_pass)}")
