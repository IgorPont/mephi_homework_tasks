from pathlib import Path

import pytest

from sessions_tasks.programming_in_python.session01.task01_reverse_blocks_text import (
    TextBlockReverser,
    reverse_blocks_text,
)


@pytest.fixture
def sample_lines():
    """
    Пример строк без пустых строк
    """
    return [
        "Привет как дела",
        "На улице идет дождь",
        "Я люблю программирование",
        "Добрый вечер друзья",
        "Сегодня отличный день",
        "Пора начинать работу",
    ]


def test_reverse_sentence():
    """
    Тест переворота слов в предложении
    """
    assert TextBlockReverser._reverse_sentence("Привет как дела") == "дела как Привет"


def test_split_fixed_size(sample_lines):
    """
    Проверка деления на блоки фиксированного размера
    """
    reverser = TextBlockReverser(block_size=3)
    result = reverser._split_fixed_size(sample_lines)
    assert result == [
        sample_lines[0:3],
        sample_lines[3:6],
    ]


def test_process_lines_no_empty(sample_lines):
    """
    Проверка обработки строк без пустых строк (по block_size)
    """
    reverser = TextBlockReverser(block_size=3)
    result = reverser.process_lines(sample_lines)
    expected = (
        "дела как Привет\n"
        "дождь идет улице На\n"
        "программирование люблю Я\n\n"
        "друзья вечер Добрый\n"
        "день отличный Сегодня\n"
        "работу начинать Пора"
    )
    assert result == expected


def test_process_lines_with_empty_lines():
    """
    Проверка, что пустые строки создают отдельные блоки
    """
    reverser = TextBlockReverser()
    lines = [
        "Привет как дела",
        "На улице идет дождь",
        "",
        "Я люблю программирование",
    ]
    result = reverser.process_lines(lines)
    expected = "дела как Привет\nдождь идет улице На\n\nпрограммирование люблю Я"
    assert result == expected


def test_process_file(tmp_path):
    """
    Проверка полной работы с реальными файлами
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    input_file = input_dir / "task01_input.txt"
    output_file = output_dir / "task01_output.txt"

    input_file.write_text(
        "Привет как дела\nНа улице идет дождь\nЯ люблю программирование\n",
        encoding="utf-8",
    )

    reverse_blocks_text(input_file, output_file, block_size=2)
    result = output_file.read_text(encoding="utf-8").strip()

    expected = "дела как Привет\nдождь идет улице На\n\nпрограммирование люблю Я"
    assert result == expected


def test_file_not_found():
    """
    Тест выбрасывания FileNotFoundError при отсутствии входного файла
    """
    reverser = TextBlockReverser()
    with pytest.raises(FileNotFoundError):
        reverser.process_file(Path("/no/such/file.txt"), Path("output.txt"))
