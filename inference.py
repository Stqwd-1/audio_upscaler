# Copyright 2026 Stanislav Suharkov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stereo audio upscaler inference with QualityController and post-processing.

Loads any audio -> processes through model -> outputs high-res WAV.
Supports:
- Judge-Jury-Executioner QC (N candidates, discriminator consensus)
- TTA (Test-Time Augmentation with phase inversion)
- Transient restoration from original
- Dynamic quantization (INT8/FP16)
"""
import torch
import torchaudio
import sys
import os
import time
import argparse
sys.path.insert(0, '.')

try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

from models.sr_network import SRNetwork
from models.spectral_unet import SpectralUNet
from models.hifi_gan import Generator as HiFiGANGenerator
from models.qc import QualityController
from models.discriminators import MultiResolutionDiscriminator
from utils.post_processing import AudioPostProcessor


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif HAS_DIRECTML:
        return torch_directml.device()
    return torch.device("cpu")


def quantize_model(model, quant_type: str = "int8"):
    """Apply dynamic quantization to a model for faster CPU inference.

    Args:
        model: PyTorch model to quantize
        quant_type: "int8" for dynamic INT8, "fp16" for half precision

    Returns:
        Quantized model
    """
    if quant_type == "int8":
        import torch.quantization
        model_cpu = model.cpu()
        quantized = torch.quantization.quantize_dynamic(
            model_cpu,
            {torch.nn.Linear, torch.nn.Conv1d, torch.nn.ConvTranspose1d},
            dtype=torch.qint8,
        )
        print(f"Quantized to INT8 (dynamic)")
        return quantized
    elif quant_type == "fp16":
        model_cpu = model.cpu()
        quantized = model_cpu.half()
        print(f"Converted to FP16")
        return quantized
    else:
        print(f"Unknown quant_type '{quant_type}', returning original model")
        return model


def load_model(checkpoint_path: str, device, stereo: bool = True):
    """Load generator + SpectralUNet + QC discriminator from checkpoint.

    Supports both SRNetwork and HiFi-GAN generators (config-driven).
    If stereo=True, forces 2-channel model even if checkpoint is mono.
    The MultiResolutionDiscriminator is loaded from the checkpoint so the
    QualityController judges use trained weights instead of random ones.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = ckpt['config']

    generator_type = config['model'].get('generator', 'sr_network')

    if generator_type == 'hifi_gan':
        hifi_cfg = config['model'].get('hifi_gan', {})
        model = HiFiGANGenerator(**hifi_cfg).to(device)
        result = model.load_state_dict(ckpt['generator'], strict=False)
        if result.missing_keys:
            print(f'Warning: missing keys in generator: {result.missing_keys}')
        if result.unexpected_keys:
            print(f'Warning: unexpected keys in generator: {result.unexpected_keys}')
        model.eval()
        print(f"HiFi-GAN Generator loaded")
    else:
        # SRNetwork
        if stereo:
            config['model']['sr_network']['in_channels'] = 2
            config['model']['sr_network']['out_channels'] = 2

        model = SRNetwork(**config['model']['sr_network']).to(device)
        state = ckpt['generator']

        # Adapt mono checkpoint to stereo
        w = state.get('input_conv.weight')
        if w is not None and w.shape[1] == 1 and stereo:
            print("Adapting mono checkpoint -> stereo (duplicating channels)")
            state['input_conv.weight'] = w.repeat(1, 2, 1)
            state['input_conv.bias'] = state.get('input_conv.bias', torch.zeros(64))
            out_w = state.get('output_conv.weight')
            if out_w is not None and out_w.shape[0] == 1:
                state['output_conv.weight'] = out_w.repeat(2, 1, 1)
                out_b = state.get('output_conv.bias')
                if out_b is not None:
                    state['output_conv.bias'] = out_b.repeat(2)

        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"SRNetwork loaded")

    # SpectralUNet
    spectral_unet = SpectralUNet(in_channels=1, base_channels=32).to(device)
    if 'spectral_unet' in ckpt:
        result = spectral_unet.load_state_dict(ckpt['spectral_unet'], strict=False)
        if result.missing_keys:
            print(f'Warning: missing keys in spectral_unet: {result.missing_keys}')
        if result.unexpected_keys:
            print(f'Warning: unexpected keys in spectral_unet: {result.unexpected_keys}')
        spectral_unet.eval()
        print("SpectralUNet loaded from checkpoint")
    else:
        print("SpectralUNet initialized (no weights in checkpoint)")

    # Discriminator for QualityController (loaded from checkpoint when present)
    discriminator = None
    if 'discriminator_spec' in ckpt:
        discriminator = MultiResolutionDiscriminator().to(device)
        discriminator.load_state_dict(ckpt['discriminator_spec'], strict=False)
        discriminator.eval()
        print("Discriminator (MRD) loaded from checkpoint for QC")

    params = sum(p.numel() for p in model.parameters())
    spec_params = sum(p.numel() for p in spectral_unet.parameters())
    print(f"Generator: {params:,} params ({params * 4 / 1024**2:.1f} MB)")
    print(f"SpectralUNet: {spec_params:,} params ({spec_params * 4 / 1024**2:.1f} MB)")

    return model, spectral_unet, config, discriminator


def _model_forward(model, chunk, spectral_unet=None, cond=None):
    """Run SRNetwork + SpectralUNet on a chunk.

    Guarantees stereo (2ch) input to model. Returns stereo output.
    """
    with torch.no_grad():
        # Ensure stereo: model always expects 2 channels
        if chunk.dim() == 3:
            if chunk.shape[1] == 1:
                chunk = chunk.repeat(1, 2, 1)
            elif chunk.shape[1] > 2:
                chunk = chunk[:, :2, :]

        pred = model(chunk, cond) if cond is not None else model(chunk)

        if spectral_unet is None:
            return pred

        # Spectral refinement: STFT -> SpectralUNet -> ISTFT
        # STFT on CPU for DirectML compatibility
        n_fft, hop, win = 1024, 256, 1024
        x = pred
        if x.dim() == 3:
            b, c, t = x.shape
            x_mono = x.reshape(b * c, t)
        else:
            x_mono = x

        x_cpu = x_mono.to("cpu")
        window = torch.hann_window(win, device="cpu", dtype=x_cpu.dtype)
        spec = torch.stft(x_cpu, n_fft, hop, win, window=window, return_complex=True)
        mag = torch.abs(spec)
        phase = torch.angle(spec)

        log_mag = torch.log(mag + 1e-7).unsqueeze(1)
        refined_log_mag = spectral_unet(log_mag.to(next(spectral_unet.parameters()).device))
        refined_mag = torch.exp(refined_log_mag.squeeze(1))

        complex_spec = refined_mag * torch.exp(1j * phase.to(refined_mag.device))
        window2 = torch.hann_window(win, device=refined_mag.device, dtype=refined_mag.dtype)
        refined_wave = torch.istft(complex_spec, n_fft, hop, win, window=window2,
                                    length=pred.shape[-1])

        # Match channels
        if refined_wave.dim() == 2:
            refined_wave = refined_wave.unsqueeze(0)
        if refined_wave.shape[0] == 1 and pred.shape[0] == 1 and pred.dim() == 3:
            refined_wave = refined_wave.expand(-1, pred.shape[1], -1)

        # Blend: mostly refined, small residual from original prediction
        final = 0.8 * refined_wave.to(pred.device) + 0.2 * pred
        return final


def upscale_audio(
    model,
    input_path: str,
    output_path: str,
    target_sr: int = 96000,
    chunk_size: int = 44100,
    device=None,
    spectral_unet=None,
    config=None,
    discriminator=None,
    chunk_overlap: int = 1024,
    use_qc: bool = False,
    n_candidates: int = 3,
    min_agreement: float = 0.7,
    use_tta: bool = False,
    transient_strength: float = 0.0,
):
    """Upscale audio file to target sample rate.

    Accepts ANY audio: mono, stereo, multi-channel, any sample rate.
    Internally converts to stereo 44100 Hz before model processing.
    Chunks overlap by ``chunk_overlap`` samples and are crossfaded so the
    seams between chunks don't introduce clicks.
    """
    if device is None:
        device = get_device()

    # Load audio
    print(f"Loading: {input_path}")
    audio, sr = torchaudio.load(input_path)
    orig_channels = audio.shape[0]
    duration = audio.shape[1] / sr
    print(f"Input: {orig_channels}ch, {sr} Hz, {duration:.1f} sec")

    # --- Normalize to stereo 44100 Hz for model ---

    # 1) Convert any channel count to stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
        print("Mono -> stereo (duplicated)")
    elif audio.shape[0] > 2:
        # Multi-channel (5.1, 7.1, etc.) -> stereo: mix down to L+R or first two channels
        left = audio[0:1, :]
        right = audio[1:2, :] if audio.shape[0] > 1 else left
        audio = torch.cat([left, right], dim=0)
        print(f"{orig_channels}ch -> stereo (took channels 1-2)")

    # 2) Resample to 44100 Hz
    if sr != 44100:
        resampler = torchaudio.transforms.Resample(sr, 44100)
        audio = resampler(audio)
        print(f"Resampled: {sr} Hz -> 44100 Hz")

    audio = audio.to(device)
    total_len = audio.shape[1]
    output_chunks = []

    # Create conditioning vector (normalized low/model sample rate ratio).
    # Uses the same ratio as training (low_sample_rate / sample_rate) so
    # the FiLM conditioning sees the same distribution at inference time.
    cond = None
    if hasattr(model, 'cond_encoder') and model.cond_encoder is not None:
        if config is not None:
            low_sr = config["data"].get("low_sample_rate", 16000)
            model_sr = config["data"].get("sample_rate", 44100)
            cond_val = low_sr / model_sr
        else:
            cond_val = sr / 44100.0
        cond = torch.full((1, 1), cond_val, device=device, dtype=audio.dtype)

    # Setup QC if enabled
    qc = None
    if use_qc:
        mrd = MultiResolutionDiscriminator().to(device)
        if discriminator is not None:
            mrd.load_state_dict(discriminator.state_dict())
            print("QC: using trained discriminator from checkpoint")
        else:
            print("QC: no discriminator in checkpoint — scoring is experimental")
        mrd.eval()
        qc = QualityController(model, discriminators=[mrd],
                               n_candidates=n_candidates, min_agreement=min_agreement,
                               forward_fn=_model_forward, spectral_unet=spectral_unet, cond=cond)
        print(f"QC enabled: {n_candidates} candidates, min_agreement={min_agreement}")

    print(f"Processing {duration:.1f} sec (stereo, 44100 Hz)...")
    start_time = time.time()

    hop = max(1, chunk_size - chunk_overlap)

    with torch.no_grad():
        pos = 0
        chunk_idx = 0
        first_chunk = True
        while pos < total_len:
            end = min(pos + chunk_size, total_len)
            chunk = audio[:, pos:end]
            input_len = chunk.shape[1]

            # Pad to multiple of 16
            pad_to = ((input_len + 15) // 16) * 16
            if chunk.shape[1] < pad_to:
                chunk = torch.nn.functional.pad(chunk, (0, pad_to - chunk.shape[1]))

            # Add batch dim
            chunk = chunk.unsqueeze(0)

            if use_qc and qc is not None:
                out = qc.upscale(chunk, verbose=True)
            elif use_tta:
                out_normal = _model_forward(model, chunk, spectral_unet, cond)
                chunk_inv = chunk * -1.0
                out_inv = _model_forward(model, chunk_inv, spectral_unet, cond)
                out = (out_normal + out_inv * -1.0) / 2.0
            else:
                out = _model_forward(model, chunk, spectral_unet, cond)

            out = out.squeeze(0)
            out = out[:, :input_len]

            # Crossfade with the previous chunk's tail so seams don't click
            if not first_chunk and chunk_overlap > 0:
                prev = output_chunks[-1]
                ov = min(chunk_overlap, input_len, prev.shape[-1])
                if ov > 0:
                    ramp = torch.linspace(0.0, 1.0, ov, device=prev.device, dtype=prev.dtype)
                    blended = prev[..., -ov:] * (1.0 - ramp) + out[..., :ov] * ramp
                    output_chunks[-1] = torch.cat(
                        [prev[..., :-ov], blended, out[..., ov:]], dim=-1
                    )
                else:
                    output_chunks.append(out.cpu())
            else:
                output_chunks.append(out.cpu())
            first_chunk = False

            pos += hop
            chunk_idx += 1
            pct = min(100, int(pos / total_len * 100))
            elapsed = time.time() - start_time
            print(f"  Chunk {chunk_idx}: {pct}% ({elapsed:.1f}s)")

    # Concatenate
    output = torch.cat(output_chunks, dim=1)
    output = output[:, :total_len]

    # Transient restoration (compare against original 44100 stereo)
    if transient_strength > 0:
        output = AudioPostProcessor.restore_transients(
            audio.cpu(), output, strength=transient_strength
        )
        print(f"Transient restoration: strength={transient_strength}")

    elapsed = time.time() - start_time
    print(f"Processing done in {elapsed:.1f} sec")

    # Resample to target sample rate
    if 44100 != target_sr:
        resampler = torchaudio.transforms.Resample(44100, target_sr)
        output = resampler(output)
        print(f"Upsampled: 44100 Hz -> {target_sr} Hz")

    # Save (always stereo output)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torchaudio.save(output_path, output.cpu(), target_sr)
    final_dur = output.shape[1] / target_sr
    print(f"Saved: {output_path}")
    print(f"Output: {output.shape[0]}ch, {target_sr} Hz, {final_dur:.1f} sec")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo audio upscaler with QC + post-processing")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output", help="Output path (default: auto)")
    parser.add_argument("-c", "--checkpoint", default="checkpoints/best_model.pt",
                        help="Model checkpoint (.pt)")
    parser.add_argument("-r", "--target-rate", type=int, default=96000)
    parser.add_argument("--chunk-size", type=int, default=44100)
    parser.add_argument("--qc", action="store_true", help="Enable QualityController (N candidates)")
    parser.add_argument("--n-candidates", type=int, default=3, help="Number of QC candidates")
    parser.add_argument("--min-agreement", type=float, default=0.7, help="QC min agreement threshold")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--transient-strength", type=float, default=0.0,
                        help="Transient restoration strength (0.0-1.0)")
    parser.add_argument("--quantize", choices=["int8", "fp16", "none"], default="none",
                        help="Dynamic quantization: int8 or fp16 (CPU only)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model, spectral_unet, config, discriminator = load_model(args.checkpoint, device)

    # Apply quantization if requested
    if args.quantize != "none" and device.type == "cpu":
        model = quantize_model(model, args.quantize)

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"outputs/{base}_{args.target_rate}hz.wav"

    upscale_audio(
        model,
        args.input,
        args.output,
        target_sr=args.target_rate,
        chunk_size=args.chunk_size,
        device=device,
        spectral_unet=spectral_unet,
        config=config,
        discriminator=discriminator,
        use_qc=args.qc,
        n_candidates=args.n_candidates,
        min_agreement=args.min_agreement,
        use_tta=args.tta,
        transient_strength=args.transient_strength,
    )
