def test_imports():
    """
    Проверяет, что основные пакеты проекта импортируются без ошибок
    """
    import homework_tasks
    import sessions_tasks
    import pytest

    assert hasattr(homework_tasks, "__package__")
    assert hasattr(sessions_tasks, "__package__")
    assert callable(pytest.main)
