# Audio Upscaler

![CI](https://github.com/Stqwd-1/audio_upscaler/actions/workflows/ci.yml/badge.svg)




Neural-network-based audio super-resolution: upscale stereo audio from 44.1 kHz to 96 / 192 / 384 kHz with GAN-based quality control.

> **Status: alpha / research preview.** Training configuration may change between
> versions, checkpoints may not be compatible across releases, and the QualityController
> and DirectML backends are experimental.

## Features

- **SRNetwork** — waveform-domain U-Net with GatedResBlocks, noise injection and optional FiLM conditioning
- **SpectralUNet** — 2-D U-Net for spectrogram refinement (complements the waveform model)
- **HiFi-GAN generator** — alternative decoder architecture
- **QualityController** — Judge-Jury-Executioner pipeline that generates several candidates and keeps only those accepted by the ensemble of discriminators *(experimental — uses the discriminator saved in the checkpoint; without one the scoring is unreliable)*
- **Discriminators** — Multi-Scale, Multi-Period and Multi-Resolution discriminators
- **Robust degradation pipeline** — simulates real-world losses on GPU (reverb, EQ, clipping, bandwidth limiting, noise) and on CPU (MP3 artifacts, quantization noise)
- **Multi-objective loss** — waveform + multi-resolution spectral + Mel + high-frequency exciter + adversarial + feature-matching
- **Streaming inference** — overlapping-chunk processing for long files without loading everything into memory at once *(the output waveform is still assembled in RAM; streaming limits the input-decoding memory)*
- **Post-processing** — TTA (phase-inversion averaging) and transient restoration
- **Backends** — CUDA (NVIDIA), CPU; DirectML (AMD/Intel) is experimental *(inference is supported, training on DirectML is best-effort — the STFT-based losses are validated on CUDA/CPU)*
- **Two UIs** — Gradio web app (`web_app.py`) and a WPF desktop app (`ui/`)

## Installation

Requirements: Python 3.10+, [ffmpeg](https://ffmpeg.org/) on PATH (needed to
decode mp3/ogg/aac/m4a/wma/opus and to run the MP3 round-trip augmentation).

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# CUDA 12.1 (train/inference on NVIDIA GPUs) — install FIRST, before the rest
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# or DirectML (Windows, AMD/Intel GPUs, experimental)
pip install torch torch-directml

# Then the project itself (installs everything from pyproject.toml)
pip install -e .

# Optional extras:
pip install -e ".[web]"      # Gradio web UI
pip install -e ".[tuning]"   # Optuna hyperparameter search
pip install -e ".[onnx]"     # ONNX export / runtime
pip install -e ".[dev]"      # tests and linting
```

See [SETUP_CUDA.md](SETUP_CUDA.md) for a step-by-step NVIDIA GPU setup guide.

## Inference

```bash
python inference.py input.wav -o output.wav -r 96000
```

Advanced pipeline (best quality, slower):

```bash
python inference.py input.wav -o output.wav --qc --tta --transient-strength 0.5
```

| Flag | Description |
|---|---|
| `-c, --checkpoint` | Path to a checkpoint (default `checkpoints/best_model.pt`) |
| `-r, --target-rate` | Target sample rate: `96000`, `192000` or `384000` |
| `--qc` | Enable QualityController (Judge-Jury-Executioner) |
| `--n-candidates` | Number of QC candidates (2–10) |
| `--tta` | Test-time augmentation (phase averaging) |
| `--transient-strength` | Transient restoration strength (0.0–1.0) |
| `--chunk-size` | Chunk size in samples (default `44100` = 1 s); chunks are overlapped by 1024 samples and crossfaded |
| `--quantize` | CPU-only dynamic quantization: `int8`, `fp16`, or `none` |

### Web UI

```bash
python web_app.py
```

### Desktop UI

Open `ui/AudioUpscaler.csproj` in Visual Studio and run (requires the Python environment and `inference.py`).

## Training

```bash
python train.py --data-dir path/to/flac --config configs/default.yaml
```

- `--data-dir` — folder with FLAC files (any nesting); required
- `--config` — `configs/default.yaml` (CPU/DirectML) or `configs/cuda.yaml` (optimized for CUDA)
- `--output-dir` — checkpoint output (default `checkpoints`)
- `--resume` — checkpoint path to continue training

Checkpoints are saved every epoch (or per `save_every` in the config) as `checkpoint_epoch_N.pt`; the best one on validation is stored as `checkpoints/best_model.pt` and used by default for inference.

Monitor training with TensorBoard:

```bash
tensorboard --logdir checkpoints/logs
```


### Обучение: лучшие практики и рекомендации

Чтобы модель действительно научилась восстанавливать высокие частоты (а не просто генерировать белый шум), следуйте этим правилам:

**1. Качество датасета (Garbage In, Garbage Out)**
*   **Формат:** Используйте только форматы без потерь (FLAC, WAV).
*   **Частота дискретизации:** Ваши исходники ДОЛЖНЫ быть в оригинальном высоком качестве (96 кГц или 192 кГц). Если вы скормите сети апскейльнутые MP3, она ничему не научится, так как там физически нет высоких частот.
*   **Без MP3-оригиналов:** Не используйте конвертированные MP3/AAC файлы как целевые (target). Во время обучения пайплайн сам портит звук (имитируя MP3-артефакты и срез частот), поэтому таргет должен быть идеальным.

**2. Объем данных**
*   **Минимум:** Для проверки (proof-of-concept) или дообучения хватит 5-10 часов качественного звука.
*   **Рекомендуется:** Для создания мощной универсальной модели нужно **от 50 до 100+ часов** музыки разных жанров, чтобы сеть не переобучилась на один инструмент.
*   Папка `data-dir` может содержать любые подпапки — скрипт рекурсивно найдет все аудиофайлы.

**3. Эпохи и время обучения**
*   Дефолтные 30 эпох в конфигурации — это просто проверка работоспособности.
*   Для датасета на 10 часов рассчитывайте на **500-1000 эпох** (примерно 500 000 – 1 000 000 шагов), пока loss-функция дискриминатора не стабилизируется.
*   Обязательно используйте `configs/cuda.yaml` и увеличьте `batch_size` (например, до 4, 8 или 16 в зависимости от VRAM вашей видеокарты), чтобы ускорить обучение.

**4. Мониторинг (TensorBoard)**
*   Запустите `tensorboard --logdir checkpoints/logs` и следите за графиками потерь.
*   **Генератор против Дискриминатора:** Это нормально, если потери дискриминатора сначала быстро падают. Со временем они должны выровняться и конкурировать. Если потери дискриминатора упали в 0.0 и не двигаются — произошел mode collapse, нужно снижать learning rate.
*   **Аудио-примеры:** Цикл обучения регулярно сохраняет примеры восстановления звука прямо в TensorBoard. Слушайте их ушами — метрики не всегда отражают реальное качество!

### Hyperparameter search

```bash
pip install -e ".[tuning]"
python tuning.py --data-dir path/to/flac
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Project structure

```
audio_upscaler/
├── models/            # SRNetwork, SpectralUNet, HiFi-GAN, discriminators, QualityController
├── data/              # dataset, degradation pipeline, augmentations, audio I/O
├── losses/            # waveform, spectral, mel, HF-exciter, adversarial, feature-matching
├── utils/             # post-processing, streaming, hardware detection, metrics
├── configs/           # training configs (default, cuda)
├── ui/                # WPF desktop application
├── tests/             # pytest smoke tests
├── .github/workflows/ # CI
├── train.py           # training entry point
├── inference.py       # inference entry point
├── web_app.py         # Gradio web UI
├── export_onnx.py     # export SRNetwork to ONNX
└── SETUP_CUDA.md      # NVIDIA GPU setup guide
```

## Acknowledgements

The generator and multi-period/multi-scale discriminators are re-implemented
following HiFi-GAN (Kong et al., 2020, "HiFi-GAN: Generative Adversarial
Networks for Efficient and High Fidelity Speech Synthesis",
[github.com/jik876/hifi-gan](https://github.com/jik876/hifi-gan), MIT License).

## License

This project is licensed under the [Apache License 2.0](LICENSE).

Copyright © 2026 Stanislav Suharkov