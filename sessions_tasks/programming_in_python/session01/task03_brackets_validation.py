"""
Задание 3. Проверка скобок.

Условие:
    Напишите функцию, которая проверяет корректность расстановки круглых, квадратных и фигурных скобок.

Описание:
    Строка считается корректной, если:
        - каждая открывающая скобка имеет соответствующую закрывающую того же типа;
        - закрывающие скобки не появляются раньше соответствующих открывающих;
        - в конце проверки стек скобок должен быть пустым.

Примеры:
    "()[]{}" -> True
    "([{}])" -> True
    "([)]" -> False
    "(((" -> False
    "abc(def)" -> True
"""


def is_brackets_balanced(s: str) -> bool:
    """
    Проверяет корректность расстановки скобок в строке.

    Аргументы:
        s (str): входная строка, содержащая произвольные символы, включая скобки.

    Возвращает:
        bool: True, если скобки расставлены корректно, иначе False.

    Алгоритм:
        - создаем словарь пар открывающих и закрывающих скобок;
        - используем стек для хранения открывающих скобок;
        - для каждого символа:
            - если он открывающий — добавляем в стек;
            - если закрывающий — проверяем соответствие последнему открытому;
        - если после обработки стек пуст — скобки корректны.

    Пример ручного вызова:
        from sessions_tasks.programming_in_python.session01.task03_brackets_validation import is_brackets_balanced
        print(is_brackets_balanced("()[]{}")) # True
        print(is_brackets_balanced("([)]")) # False
    """
    # ключ - закрывающая скобка, значение - открывающая скобка
    pairs: dict[str, str] = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []

    for char in s:
        # открывающая скобка
        if char in pairs.values():
            stack.append(char)
        # закрывающая скобка
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()

    return not stack


if __name__ == "__main__":
    """
    Запуск через poetry:
        poetry run python sessions_tasks/programming_in_python/session01/task03_brackets_validation.py
    """
    examples = [
        "()[]{}",
        "([{}])",
        "([)]",
        "(((",
        "abc(def)",
        "a + (b * [c - {d / e}])",
        "((a+b]",
    ]

    print("🧩 Результат работы проверки скобок:\n")
    for text in examples:
        print(f"{text:25} -> {is_brackets_balanced(text)}")
