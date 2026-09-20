import torch
import torchaudio
import pytest
from pathlib import Path

from models.sr_network import SRNetwork
from inference import load_model, upscale_audio
from utils.streaming import StreamingProcessor

@pytest.fixture
def mock_wav(tmp_path):
    wav_path = tmp_path / "mock.wav"
    # Create 0.5s stereo audio at 44100Hz
    audio = torch.randn(2, 22050)
    torchaudio.save(wav_path, audio, 44100)
    return wav_path

@pytest.fixture
def mock_checkpoint(tmp_path):
    ckpt_path = tmp_path / "mock_ckpt.pt"
    generator = SRNetwork(in_channels=2, out_channels=2, nf=16)
    config = {
        "model": {"generator": "sr_network", "sr_network": {"nf": 16}},
        "data": {"sample_rate": 44100}
    }
    torch.save({"generator": generator.state_dict(), "config": config}, ckpt_path)
    return ckpt_path

def test_inference_cpu(mock_wav, mock_checkpoint, tmp_path):
    out_path = tmp_path / "out.wav"
    device = torch.device("cpu")
    model, spectral_unet, config, _ = load_model(str(mock_checkpoint), device=device)
    
    upscale_audio(
        input_path=str(mock_wav),
        output_path=str(out_path),
        model=model,
        target_sr=96000,
        chunk_size=44100,
        device=device,
        spectral_unet=spectral_unet,
        config=config,
    )
    
    assert out_path.exists()
    out_audio, out_sr = torchaudio.load(out_path)
    assert out_sr == 96000
    assert out_audio.shape[0] == 2
    assert torch.isfinite(out_audio).all()

def test_streaming_cpu(mock_wav, mock_checkpoint, tmp_path):
    out_path = tmp_path / "out_stream.wav"
    device = torch.device("cpu")
    model, spectral_unet, config, _ = load_model(str(mock_checkpoint), device=device)
    
    processor = StreamingProcessor(model, chunk_size=44100, overlap=1024, spectral_unet=spectral_unet, cond=None)
    processor.process_file(
        str(mock_wav),
        str(out_path),
        target_sr=96000,
    )
    
    assert out_path.exists()
    out_audio, out_sr = torchaudio.load(out_path)
    assert out_sr == 96000
    assert out_audio.shape[0] == 2
    assert torch.isfinite(out_audio).all()
