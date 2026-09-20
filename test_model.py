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

"""End-to-end model inference test on a real audio file.

Usage:
    python test_model.py input.mp3 --output outputs/test_result.wav --checkpoint checkpoints/best_model.pt

Processes the file in ~1 second chunks and saves the upscaled result.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(__file__))

from models.sr_network import SRNetwork
from data.formats import _resample, save_audio


def main():
    parser = argparse.ArgumentParser(description="Test SRNetwork on a real audio file")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output", default="outputs/test_result.wav")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--target-rate", type=int, default=96000)
    args = parser.parse_args()

    try:
        import torch_directml
        device = torch_directml.device()
        device_name = torch_directml.device_name(0)
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = str(device)
    print(f"Device: {device_name}")

    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ckpt["config"]

    model = SRNetwork(**config["model"]["sr_network"]).to(device)
    model.load_state_dict(ckpt["generator"], strict=False)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    print(f"Loading: {args.input}")
    audio, sr = torchaudio.load(args.input)
    print(f"Input: {audio.shape}, {sr} Hz, {audio.shape[1]/sr:.1f} sec, channels: {audio.shape[0]}")

    model_sr = config["data"]["sample_rate"]
    if sr != model_sr:
        resampler = torchaudio.transforms.Resample(sr, model_sr)
        audio = resampler(audio)
        print(f"Resampled to {model_sr} Hz")

    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]

    audio = audio.unsqueeze(0).to(device)
    print(f"Input shape: {audio.shape} (batch, channels, samples)")

    chunk_size = model_sr
    total_len = audio.shape[-1]
    output_chunks = []

    print(f"Processing {total_len/model_sr:.1f} sec in {chunk_size/model_sr:.1f}s chunks...")
    start_time = time.time()

    with torch.no_grad():
        pos = 0
        chunk_idx = 0
        while pos < total_len:
            end = min(pos + chunk_size, total_len)
            chunk = audio[..., pos:end]
            input_len = chunk.shape[-1]

            pad_to = ((input_len + 15) // 16) * 16
            if chunk.shape[-1] < pad_to:
                chunk = torch.nn.functional.pad(chunk, (0, pad_to - chunk.shape[-1]))

            out = model(chunk)
            out = out[..., :input_len]
            output_chunks.append(out.cpu())
            pos += chunk_size
            chunk_idx += 1

            elapsed = time.time() - start_time
            pct = min(100, int(pos / total_len * 100))
            print(f"  Chunk {chunk_idx}: {pct}% ({elapsed:.1f}s)")

    output = torch.cat(output_chunks, dim=-1)
    output = output[..., :total_len]
    print(f"Done in {time.time() - start_time:.1f} sec")

    output_np = output.squeeze(0).cpu().numpy()
    output_np = output_np.T

    max_val = np.max(np.abs(output_np))
    if max_val > 1.0:
        output_np = output_np / max_val

    if args.target_rate != model_sr:
        print(f"Resampling {model_sr}Hz -> {args.target_rate}Hz")
        output_np = _resample(output_np, model_sr, args.target_rate)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_audio(args.output, output_np, args.target_rate)

    ch_str = "stereo" if output_np.ndim > 1 and output_np.shape[1] > 1 else "mono"
    print(f"Saved: {args.output}")
    print(f"Output: {output_np.shape}, {args.target_rate} Hz, {output_np.shape[0]/args.target_rate:.1f} sec, {ch_str}")


if __name__ == "__main__":
    main()