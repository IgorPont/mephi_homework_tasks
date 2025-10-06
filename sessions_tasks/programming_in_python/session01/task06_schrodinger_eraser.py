"""
Задание 6. Задача Шредингера — "стирание" фрагментов текста.

Условие:
    Пользователь указывает файл и процент текста, который нужно удалить (например, 30%).
    Программа случайно выбирает слова или части абзацев и заменяет их на пробелы или точки,
    сохраняя общий объем документа (длина строки не меняется).

Пример:
    Вход:
        "Сегодня солнечный день, и я собираюсь гулять в парке с моими друзьями."
    Выход:
        "Сегодня .......... день, и я собираюсь гулять .. ...... с моими друзьями."

Замечания:
    - Берется процент относительно общего числа символов исходной строки.
    - Разделители (пробелы, запятые, точки) сохраняются, как в исходном файле.
"""

import random
import re
from pathlib import Path

# Регулярное выражение для поиска слов любых языков
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def erase_random_words(text: str, percent: int) -> str:
    """
    Удаляет случайные слова из текста, заменяя их на троеточие ("...").

    Аргументы:
        text (str): исходный текст.
        percent (int): процент слов, которые нужно "стереть" (0–100).

    Возвращает:
        str: текст, в котором случайные слова заменены на '...'.
    """
    # Находим все слова по позиции (start - первый символ слова, end - последний символ слова),
    # записываем "координаты" в список кортежей
    words = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    if not words:
        return text

    # Количество слов, которые нужно стереть (минимум одно слово будет стерто)
    erase_count = max(1, len(words) * percent // 100)
    # Рандомно по координатам выбираем слова, которые нужно стереть
    to_erase = random.sample(words, k=erase_count)

    # Разбиваем текст на список символов
    chars = list(text)
    # Пробегаемся по координатам слов, которые нужно стереть
    for start, end in to_erase:
        # Заменяем символы выбранного слова на точки
        for i in range(start, end):
            chars[i] = "."

    # Собираем итоговую строку
    return "".join(chars)


def erase_in_file(input_file: str | Path, output_file: str | Path, percent: int) -> None:
    """
    Читает текстовый файл, применяет "стирание" и сохраняет результат.

    Аргументы:
        input_file (str | Path): путь к входному файлу.
        output_file (str | Path): путь к файлу, куда сохранить результат.
        percent (int): процент текста, который нужно удалить.

    Вызов функции напрямую:
        erase_in_file("input.txt", "output.txt", percent=30)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Файл '{input_file}' не найден.")

    text = input_path.read_text(encoding="utf-8")
    # Стираем слова
    processed = erase_random_words(text, percent)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Записываем результат в новый файл
    output_path.write_text(processed, encoding="utf-8")


if __name__ == "__main__":
    """
    Пример запуска через Poetry:
        poetry run python sessions_tasks/programming_in_python/session01/task06_schrodinger_eraser.py
    """

    erase_in_file(
        input_file="sessions_tasks/programming_in_python/session01/input/task06_input.txt",
        output_file="sessions_tasks/programming_in_python/session01/output/task06_output.txt",
        percent=30,
    )
    print(" ✅ Эффект Шредингера применен!\nРезультат сохранен в output/task06_output.txt")
