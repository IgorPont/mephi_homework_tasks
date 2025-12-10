"""
Скрипт для удобного запуска проекта UrbanEcoSoundMonitor.

Задачи скрипта:
    1. Установить зависимости из requirements.txt (через текущий интерпретатор Python).
    2. Запустить выбранный режим:
       - streamlit -> веб-интерфейс для демонстрации;
       - notebook -> Jupyter-ноутбук с полным решением.

Пример использования (внутри папки UrbanEcoSoundMonitor):
    python run_project.py --mode streamlit
"""

import argparse
import subprocess
import sys
from pathlib import Path


def install_requirements(requirements_path: Path) -> None:
    """
    Устанавливает зависимости из файла requirements.txt.

    Параметры:
        requirements_path: Path
            Путь до файла requirements.txt.

    Функция использует текущий интерпретатор Python (sys.executable) и
    выполняет команду:
        python -m pip install -r requirements.txt

    Если файл requirements.txt отсутствует, просто выводим предупреждение
    и пропускаем установку (на случай, если зависимости уже стоят).
    """
    if not requirements_path.exists():
        print(f"[!] Файл {requirements_path} не найден, пропускаю установку зависимостей")
        return

    print(f"[+] Устанавливаю зависимости из {requirements_path} ...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
        )
        print("[+] Зависимости успешно установлены")
    except subprocess.CalledProcessError as exc:
        print("[!] Ошибка при установке зависимостей")
        print(f"    Код возврата: {exc.returncode}")
        sys.exit(exc.returncode)


def run_streamlit_app(project_root: Path) -> None:
    """
    Запускает Streamlit-приложение.

    Параметры:
        project_root: Path
            Корень пакета UrbanEcoSoundMonitor (папка, где лежит streamlit_app.py).
    """
    app_path = project_root / "streamlit_app.py"
    if not app_path.exists():
        print(f"[!] Не найден файл {app_path}")
        sys.exit(1)

    print("[+] Запускаю Streamlit-приложение...")
    # Команда: streamlit run streamlit_app.py
    subprocess.call(["streamlit", "run", str(app_path)])


def run_notebook(project_root: Path) -> None:
    """
    Запускает Jupyter Lab с основным решением.

    Параметры:
        project_root: Path
            Корневая папка пакета UrbanEcoSoundMonitor.
    """
    notebook_path = project_root / "urban_ecosystem_sound_monitoring.ipynb"
    if not notebook_path.exists():
        print(f"[!] Не найден ноутбук {notebook_path}")
        sys.exit(1)

    print("[+] Запускаю Jupyter Lab...")
    subprocess.call(["jupyter", "lab", str(notebook_path)])


def parse_args() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.

    Поддерживаемые флаги:
        -m / --mode: streamlit | notebook
    """
    parser = argparse.ArgumentParser(
        description="Запуск учебного проекта UrbanEcoSoundMonitor"
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=("streamlit", "notebook"),
        default="streamlit",
        help="Режим запуска: 'streamlit' (по умолчанию) или 'notebook'",
    )
    return parser.parse_args()


def main() -> None:
    """
    Точка входа скрипта.

    Логика:
        1. Определяет корень пакета (папка с этим файлом).
        2. Устанавливает зависимости из requirements.txt.
        3. Запускает выбранный режим (Streamlit или Notebook).
    """
    project_root = Path(__file__).resolve().parent
    requirements_path = project_root / "requirements.txt"

    args = parse_args()

    # 1. Устанавливаем зависимости
    install_requirements(requirements_path)

    # 2. Запускаем выбранный режим
    if args.mode == "streamlit":
        run_streamlit_app(project_root)
    else:
        run_notebook(project_root)


if __name__ == "__main__":
    main()
