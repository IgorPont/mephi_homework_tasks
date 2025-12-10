"""
Генерация тестовых аудио-файлов для UrbanEcoSoundMonitor.

Создает в папке data/audio_samples для последующего анализа три тестовых аудио-файла:
    - example_tone_440hz.wav -> чистый тон 440 Гц
    - example_city_noise.wav -> смесь низкого гула и шума (городской фон)
    - example_siren_sweep.wav -> простая сирена (частотный свип)

Пример запуска:
    # Запуск должен быть из директории UrbanEcoSoundMonitor
    python generate_test_audio.py
"""

from pathlib import Path
import numpy as np
import wave

SAMPLE_RATE = 16000
DURATION_SEC = 3.0


def _write_wav(path: Path, data: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """
    Записывает моно WAV 16-bit PCM.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    max_val = np.max(np.abs(data)) or 1.0
    scaled = (data / max_val * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(scaled.tobytes())


def main() -> None:
    base_dir = Path(__file__).resolve().parent / "data" / "audio_samples"
    base_dir.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0, DURATION_SEC, int(SAMPLE_RATE * DURATION_SEC), endpoint=False)

    # 1. Чистый тон 440 Гц
    tone = 0.8 * np.sin(2 * np.pi * 440 * t)
    _write_wav(base_dir / "example_tone_440hz.wav", tone)

    # 2. "Городской шум" — низкий гул + белый шум
    noise = 0.4 * np.random.randn(len(t))
    low_hum = 0.6 * np.sin(2 * np.pi * 70 * t)
    city_noise = noise + low_hum
    _write_wav(base_dir / "example_city_noise.wav", city_noise)

    # 3. "Сирена" — частотный свип
    f_start, f_end = 600, 1200
    freq_sweep = np.linspace(f_start, f_end, len(t))
    siren = 0.8 * np.sin(2 * np.pi * freq_sweep * t)
    _write_wav(base_dir / "example_siren_sweep.wav", siren)

    print(f"[+] Сгенерированы тестовые файлы в {base_dir}")


if __name__ == "__main__":
    # Импорт здесь, чтобы не тащить в глобальный scope лишнее
    import numpy as np

    main()
