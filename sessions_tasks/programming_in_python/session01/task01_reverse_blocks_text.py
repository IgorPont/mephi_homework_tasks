"""
Задание 1.

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
        # размер блока с предложениями, по дефолту = 3
        self.block_size = block_size

    @staticmethod
    def _reverse_sentence(sentence: str) -> str:
        """
        Переворачивает слова внутри предложения.
        Возвращает строку с перевернутыми словами.
        """
        words = sentence.strip().split()
        return " ".join(reversed(words))

    def _split_by_empty_lines(self, lines: list[str]) -> list[list[str]]:
        """
        Разделяет строки на блоки по пустым строкам.
        Возвращает список со списками блоков по строкам.
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
        Делит список строк на блоки фиксированного размера.
        Возвращает список со списками строк, разделенными на блоки установленного размера.

        Пример:
            при block_size = 3 и входных строках:
            ["a", "b", "c", "d", "e", "f", "g"]

            результат:
            [
                ["a", "b", "c"],
                ["d", "e", "f"],
                ["g"]
            ]
        """
        blocks: list[list[str]] = []
        total_lines = len(lines)

        # Идем по списку с шагом размера блока
        for start_index in range(0, total_lines, self.block_size):
            end_index = start_index + self.block_size
            block = lines[start_index:end_index]
            blocks.append(block)

        return blocks

    def process_lines(self, lines: list[str]) -> str:
        """
        Обрабатывает список строк и возвращает текст:
            - Пустые строки используются как разделители блоков (если есть);
            - Либо деление происходит по block_size.
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
        Читает файл, обрабатывает и сохраняет результат
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
    Запуск напрямую через функцию-обертку.
    Пример запуска:
        `reverse_blocks_text("input.txt", "output.txt")`
    """
    reverser = TextBlockReverser(block_size=block_size)
    reverser.process_file(Path(input_file), Path(output_file))


if __name__ == "__main__":
    """
    Запуск через poetry:
        `poetry run python sessions_tasks/programming_in_python/session01/task01_reverse_blocks_text.py`
    """
    reverse_blocks_text(
        input_file="sessions_tasks/programming_in_python/session01/input/task01_input.txt",
        output_file="sessions_tasks/programming_in_python/session01/output/task01_output.txt",
    )
    print(" ✅ Успех!\nРезультат записан в output/task01_output.txt")
