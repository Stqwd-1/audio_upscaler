# Audio Upscaler GUI (WPF)

Desktop GUI wrapper for `inference.py`.

## Requirements

- .NET 8.0 SDK: https://dotnet.microsoft.com/download/dotnet/8.0
- Python 3.10+ with the audio_upscaler venv activated

## Build & Run

```bash
cd ui
dotnet build -c Release
dotnet run -c Release
```

Or publish as a standalone executable:

```bash
dotnet publish -c Release -r win-x64 --self-contained
```

The output will be in `ui/bin/Release/net8.0-windows/win-x64/publish/`.

## Features

- File browser for input audio, checkpoint, and output directory
- Target sample rate selection (96k / 192k / 384k)
- Quality Controller toggle with candidate count and agreement sliders
- TTA, Transient Restoration options
- INT8 / FP16 dynamic quantization
- Real-time console output from inference.py
