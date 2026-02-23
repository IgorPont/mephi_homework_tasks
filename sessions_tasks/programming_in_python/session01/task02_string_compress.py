"""
Задание 2. Сжатие строки.

Условие:
    Напишите функцию, которая принимает строку и сжимает ее определенным образом.
    Строки содержат произвольные символы, включая пробелы и спецсимволы, и требуют точного учета длины при кодировании.
    Сжатие строки происходит сериями одинаковых символов в формате символ+количество,
    но только если длина сжатой строки строго меньше исходной.

Примеры:
    вход: "aaabbc" -> вывод: "a3b2c"
    вход: "abcd" -> вывод: "abcd" (так как сжатая не короче)
    вход: "111112222233" -> вывод: "152532"
"""


def compress_string(s: str) -> str:
    """
    Сжимает строку, заменяя последовательности одинаковых символов на символ + количество повторений.

    Аргументы:
        s (str): исходная строка (может содержать любые символы, включая пробелы и спецсимволы).

    Возвращает:
        str: сжатая строка, если она короче исходной, иначе возвращает исходную строку.

    Алгоритм:
        - Проходим строку посимвольно;
        - Для каждой группы одинаковых символов добавляем "<символ><количество>", если количество > 1;
        - Если количество = 1, добавляем только символ;
        - В конце сравниваем длину сжатой и исходной строк, возвращаем более короткий вариант.

    Примеры:
        вход: "aaabbc" -> вывод: "a3b2c"
        вход: "abcd" -> вывод: "abcd" (так как сжатая не короче)
        вход: "111112222233" -> вывод: "152532"

    Пример ручного вызова:
        from sessions_tasks.programming_in_python.session01.task02_string_compress import compress_string
        print(compress_string("aaabbc")) # a3b2c
        print(compress_string("abcd")) # abcd
        print(compress_string("111112222233")) # 152532
    """
    if not s:
        return s

    parts: list[str] = []
    current_char = s[0]
    count = 1

    for ch in s[1:]:
        if ch == current_char:
            count += 1
        else:
            # добавляем завершенную серию повторяющихся символов
            parts.append(current_char if count == 1 else f"{current_char}{count}")
            current_char = ch
            count = 1

    # добавляем последнюю серию
    parts.append(current_char if count == 1 else f"{current_char}{count}")

    # объединяем список серий в итоговую строку
    compressed = "".join(parts)

    # возвращаем, если сжатая строка короче исходной
    return compressed if len(compressed) < len(s) else s


if __name__ == "__main__":
    """
    Запуск через Poetry:
        poetry run python sessions_tasks/programming_in_python/session01/task02_string_compress.py
    """
    examples = ["aaabbc", "abcd", "", "    ", "!!**", "aabccccaaa", "111112222233"]

    print("🧩 Результат работы compress_string():\n")
    for text in examples:
        print(f"Вход: {text!r} -> Выход: {compress_string(text)}")
