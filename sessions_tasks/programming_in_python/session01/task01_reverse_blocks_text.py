"""
Задание 1.
Обратный порядок слов в блоках текста.

Условие:
    Дан текстовый файл, каждое предложение которого занимает одну строку.
    Напишите программу, которая разделяет текст на блоки — каждый блок состоит из нескольких предложений.
    Затем переворачивает порядок слов только внутри каждого предложения, не меняя порядок самих предложений в блоке.

    Файл содержит:
    1. Привет как дела
    2. На улице идет дождь
    3. Я люблю программирование
"""

from pathlib import Path


class TextBlockReverser:
    """
    Класс для обработки текстовых файлов:
        - Делит текст на блоки;
        - Переворачивает слова в каждом предложении;
        - Сохраняет порядок предложений в блоке.
    """

    def __init__(self, block_size: int = 3) -> None:
        """
        Инициализирует экземпляр класса TextBlockReverser.

        Аргументы:
            block_size (int): Количество предложений в одном блоке (по умолчанию — 3).
        """
        self.block_size = block_size

    @staticmethod
    def _reverse_sentence(sentence: str) -> str:
        """
        Переворачивает порядок слов внутри одного предложения.

        Аргументы:
            sentence (str): Исходное предложение.

        Возвращает:
            str: Предложение с перевернутым порядком слов.
        """
        words = sentence.strip().split()
        return " ".join(reversed(words))

    def _split_by_empty_lines(self, lines: list[str]) -> list[list[str]]:
        """
        Разделяет текст на блоки по пустым строкам.

        Аргументы:
            lines (list[str]): Список строк текста, где каждая строка — отдельное предложение.

        Возвращает:
            list[list[str]]: Список блоков, каждый из которых представляет собой список строк (предложений).
        """
        blocks: list[list[str]] = []
        current_block: list[str] = []

        for line in lines:
            if line.strip() == "":
                if current_block:
                    blocks.append(current_block)
                    current_block = []
            else:
                current_block.append(line)

        if current_block:
            blocks.append(current_block)

        return blocks

    def _split_fixed_size(self, lines: list[str]) -> list[list[str]]:
        """
        Делит текст на блоки фиксированного размера (по числу предложений).

        Аргументы:
            lines (list[str]): Список строк текста, где каждая строка — отдельное предложение.

        Возвращает:
            list[list[str]]: Список блоков, разделенных по количеству предложений.

        Пример:
            При block_size = 3 и входных строках:
                ["a", "b", "c", "d", "e", "f", "g"]

            Результат:
                [
                    ["a", "b", "c"],
                    ["d", "e", "f"],
                    ["g"]
                ]
        """
        blocks: list[list[str]] = []
        total_lines = len(lines)

        for start_index in range(0, total_lines, self.block_size):
            end_index = start_index + self.block_size
            block = lines[start_index:end_index]
            blocks.append(block)

        return blocks

    def process_lines(self, lines: list[str]) -> str:
        """
        Обрабатывает список строк и возвращает результат.

        Аргументы:
            lines (list[str]): Список строк исходного текста.

        Возвращает:
            str: Текст, где в каждом предложении слова перевернуты, а порядок предложений сохранен.
                 Пустые строки используются как разделители блоков (если есть),
                 либо деление выполняется по параметру block_size.
        """
        has_empty = any(line.strip() == "" for line in lines)
        blocks = self._split_by_empty_lines(lines) if has_empty else self._split_fixed_size(lines)

        result_blocks: list[str] = []
        for block in blocks:
            reversed_sentences = [self._reverse_sentence(sentence) for sentence in block if sentence.strip()]
            result_blocks.append("\n".join(reversed_sentences))

        return "\n\n".join(result_blocks)

    def process_file(self, input_path: Path, output_path: Path) -> None:
        """
        Обрабатывает текстовый файл, разделяя его на блоки и переворачивая слова в предложениях.

        Аргументы:
            input_path (Path): Путь к входному файлу с исходным текстом.
            output_path (Path): Путь к выходному файлу, куда будет записан результат.

        Исключения:
            FileNotFoundError: Если входной файл не найден.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Файл {input_path} не найден")

        with input_path.open(encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f]

        result = self.process_lines(lines)

        with output_path.open("w", encoding="utf-8") as f:
            f.write(result)


def reverse_blocks_text(input_file: str | Path, output_file: str | Path, block_size: int = 3) -> None:
    """
    Функция-обертка для быстрого запуска обработки текстового файла.

    Аргументы:
        input_file (str | Path): Путь к входному файлу.
        output_file (str | Path): Путь к выходному файлу.
        block_size (int): Количество предложений в блоке (по умолчанию — 3).

    Пример использования:
        reverse_blocks_text("input.txt", "output.txt")
    """
    reverser = TextBlockReverser(block_size=block_size)
    reverser.process_file(Path(input_file), Path(output_file))


if __name__ == "__main__":
    """
    Пример запуска через Poetry:
        poetry run python sessions_tasks/programming_in_python/session01/task01_reverse_blocks_text.py
    """
    reverse_blocks_text(
        input_file="sessions_tasks/programming_in_python/session01/input/task01_input.txt",
        output_file="sessions_tasks/programming_in_python/session01/output/task01_output.txt",
    )
    print(" ✅ Успех!\nРезультат записан в output/task01_output.txt")
