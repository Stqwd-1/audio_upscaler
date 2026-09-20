# Audio Upscaler — Установка и запуск на NVIDIA GPU

## Требования

- Python 3.10+
- NVIDIA GPU с поддержкой CUDA (8GB+ VRAM рекомендуется)
- Windows или Linux
- [ffmpeg](https://ffmpeg.org/) в PATH (нужен для декодирования mp3/ogg/aac/m4a/wma/opus и MP3-аугментации)

Примечание: обучение и инференс на CPU и DirectML (AMD/Intel) возможны, но
экспериментальны — STFT-потери проверены на CUDA. Для серьёзного обучения
рекомендуется NVIDIA GPU.

## 1. Установка Python

Скачать Python 3.10+ с https://www.python.org/downloads/

При установке поставить галочку "Add Python to PATH".

## 2. Создание виртуального окружения

```bash
cd audio_upscaler
python -m venv .venv
```

Активация:
```bash
# Windows
.\.venv\Scripts\activate

# Linux
source .venv/bin/activate
```

## 3. Установка PyTorch с CUDA

ВАЖНО: Установить PyTorch ПЕРВЫМ, до остальных зависимостей.

```bash
# CUDA 12.1 (рекомендуется)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Или CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Проверка:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Должно вывести: `CUDA: True, Device: NVIDIA GeForce RTX ...`

## 4. Установка остальных зависимостей

```bash
pip install -e .
```

Опционально (веб-интерфейс, поиск гиперпараметров, ONNX):

```bash
pip install -e ".[web]"
pip install -e ".[tuning]"
pip install -e ".[onnx]"
```

> Используйте именно `pip install -e .`, а не ручной список пакетов — зависимости описаны в `pyproject.toml` (torch/torchaudio ставятся отдельно, см. шаг 3).

## 5. Подготовка данных

Датасет должен быть в папке с FLAC файлами (любая вложенность).

Пример структуры:
```
my_dataset/
├── album1/
│   ├── track01.flac
│   └── track02.flac
├── album2/
│   └── track01.flac
└── loose_track.flac
```

Если данные в MP3 — конвертируйте в WAV:
```bash
ffmpeg -i track.mp3 -ar 44100 track.wav
```

## 6. Запуск тренировки

```bash
# CUDA конфиг (рекомендуется)
python train.py --data-dir путь/к/данным --config configs/cuda.yaml

# Или дефолтный конфиг
python train.py --data-dir путь/к/данным --config configs/default.yaml
```

### Параметры запуска

| Параметр | Описание | По умолчанию |
|---|---|---|
| `--data-dir` | Папка с FLAC файлами | (обязательный) |
| `--config` | Конфигурация | `configs/default.yaml` |
| `--output-dir` | Папка для чекпоинтов | `checkpoints` |
| `--resume` | Путь к чекпоинту для продолжения | нет |

### Настройка batch_size (в конфиге)

Зависит от VRAM вашей GPU:

| VRAM | batch_size | gradient_accumulation |
|---|---|---|
| 6 GB | 2 | 4 |
| 8 GB | 4 | 2 |
| 12 GB | 6 | 2 |
| 16 GB | 8 | 1 |
| 24 GB | 12 | 1 |

Если OOM (ошибка нехватки памяти) — уменьшите batch_size.

## 7. Мониторинг тренировки

```bash
# В отдельном терминале
tensorboard --logdir checkpoints/logs
```

Откройте http://localhost:6006 в браузере.

## 8. Инференс (получение результата)

```bash
# Базовый
python inference.py input.wav -o output.wav -r 96000

# С QualityController (лучшее качество, медленнее)
python inference.py input.wav -o output.wav --qc --n-candidates 5

# С TTA (усреднение фазы)
python inference.py input.wav -o output.wav --tta

# Полный пайплайн
python inference.py input.wav -o output.wav --qc --tta --transient-strength 0.5
```

### Параметры инференса

| Параметр | Описание |
|---|---|
| `-c` / `--checkpoint` | Путь к чекпоинту (по умолчанию `checkpoints/best_model.pt`) |
| `-r` / `--target-rate` | Целевая ЧД (96000/192000/384000) |
| `--chunk-size` | Размер чанка (44100 = 1 сек) |
| `--qc` | Включить QualityController |
| `--n-candidates` | Кол-во кандидатов QC (2-10) |
| `--tta` | Test-Time Augmentation |
| `--transient-strength` | Восстановление транзиентов (0.0-1.0) |
| `--quantize` | Динамическое квантование на CPU: `int8` или `fp16` |

## 9. Структура проекта

```
audio_upscaler/
├── models/
│   ├── sr_network.py       # SRNetwork (U-Net, waveform domain)
│   ├── spectral_unet.py     # SpectralUNet (2D U-Net, spectrograms)
│   ├── discriminators.py    # 3 дискриминатора (MSD, MPD, MRD)
│   ├── qc.py               # QualityController (Judge-Jury-Executioner)
│   └── hifi_gan.py          # HiFi-GAN (запасная архитектура)
├── data/
│   ├── dataset.py           # AudioDataset
│   ├── degradation.py       # AdvancedDegradation (GPU)
│   ├── augmentations.py     # AudioAugmentations (NumPy)
│   ├── transforms.py        # CPU-трансформы
│   └── formats.py           # Аудио I/O
├── losses/
│   ├── spectral.py          # SpectralLoss, LSDLoss
│   ├── mel_loss.py          # MelLoss
│   └── perceptual.py        # AdversarialLoss, FeatureMatchingLoss
├── utils/
│   ├── post_processing.py   # Transient Restore
│   ├── hardware.py          # Device detection
│   └── metrics.py           # LSD, SSIM
├── configs/
│   ├── default.yaml         # Дефолтный конфиг (DirectML/CPU)
│   └── cuda.yaml            # CUDA-оптимизированный конфиг
├── tests/                   # pytest smoke tests
├── train.py                 # Тренировка
├── inference.py             # Инференс
├── web_app.py               # Gradio UI
└── SETUP_CUDA.md            # Эта инструкция
```

## 10. Чекпоинты

Чекпоинты сохраняются в `checkpoints/` по расписанию `save_every` из конфига.

Формат: `checkpoint_epoch_N.pt` — периодические снимки; лучший на валидации набор весов сохраняется как `best_model.pt` (используется по умолчанию в инференсе).

Содержит:
- `generator` — веса SRNetwork
- `spectral_unet` — веса SpectralUNet
- `discriminator_s/p/spec` — веса дискриминаторов
- `opt_g/opt_d` — состояния оптимизаторов
- `config` — полная конфигурация

## Частые проблемы

### OOM (Out of Memory)
Уменьшите `batch_size` в конфиге. Для 8GB VRAM: `batch_size: 2`.

### Медленная тренировка
Убедитесь что `mixed_precision: true` в конфиге. Увеличьте `num_workers`.

### torch.stft ошибка
Убедитесь что установили PyTorch с CUDA, а не CPU-only версию.
