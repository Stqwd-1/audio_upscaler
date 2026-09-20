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

"""Streaming processor for long audio files.

Loads and processes audio chunk-by-chunk (torchaudio.load with
frame_offset/num_frames), so files longer than a few minutes never get
decoded into memory in one piece. Chunks overlap and are crossfaded to
avoid clicks at the seams.

Note: streaming reduces the input-decoding memory, but the full output
waveform is still assembled in RAM (model activations + one stereo float
array). For truly disk-resident processing a wave-based output writer
would be needed instead.
"""
import os

import torch
import torch.nn.functional as F
import torchaudio


class StreamingProcessor:
    """Process long audio files in overlapping chunks, stereo only.

    Args:
        model: Generator model (stereo in / stereo out).
        spectral_unet: SpectralUNet (optional spectral refinement).
        chunk_size: Samples per chunk at 44100 Hz (default: 1 sec).
        overlap: Samples of crossfade between chunks at 44100 Hz.
        device: Processing device.
    """
    def __init__(
        self,
        model,
        spectral_unet=None,
        chunk_size: int = 44100,
        overlap: int = 1024,
        device=None,
    ):
        self.model = model
        self.spectral_unet = spectral_unet
        self.chunk_size = chunk_size
        self.overlap = min(overlap, chunk_size // 2)
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def process_file(
        self,
        input_path: str,
        output_path: str,
        target_sr: int = 96000,
        verbose: bool = True,
    ) -> str:
        """Process a long audio file with streaming.

        Returns: output_path
        """
        info = torchaudio.info(input_path)
        src_sr = info.sample_rate
        num_channels = info.num_channels
        total_frames = info.num_frames

        if verbose:
            print(f"Streaming: {num_channels}ch, {src_sr} Hz, {total_frames / src_sr:.1f} sec")

        # Work in a 44100 Hz domain internally, then resample once at the end.
        step = self.chunk_size - self.overlap
        ratio = src_sr / 44100.0
        chunks_44100_end = int(total_frames * 44100 / src_sr + 0.5)

        resampler_in = torchaudio.transforms.Resample(src_sr, 44100)
        if 44100 != target_sr:
            resampler_out = torchaudio.transforms.Resample(44100, target_sr)
        else:
            resampler_out = None

        fade = torch.linspace(0.0, 1.0, self.overlap) if self.overlap > 0 else None

        segments = []
        chunk_idx = 0
        pos_44100 = 0

        while pos_44100 < chunks_44100_end:
            # Map the 44100-domain position back to source frames for a lazy disk read.
            src_start = int(pos_44100 * ratio)
            src_end = min(total_frames, int((pos_44100 + self.chunk_size) * ratio) + 1)
            chunk, _ = torchaudio.load(
                input_path, frame_offset=src_start, num_frames=src_end - src_start
            )

            # To stereo.
            if chunk.shape[0] == 1:
                chunk = chunk.repeat(2, 1)
            elif chunk.shape[0] > 2:
                chunk = chunk[:2, :]

            if src_sr != 44100:
                chunk = resampler_in(chunk)

            # Normalize to exactly chunk_size samples at 44100 Hz.
            if chunk.shape[1] < self.chunk_size:
                chunk = F.pad(chunk, (0, self.chunk_size - chunk.shape[1]))
            elif chunk.shape[1] > self.chunk_size:
                chunk = chunk[:, :self.chunk_size]

            # Add batch dim and process.
            out = self._model_forward(chunk.unsqueeze(0).to(self.device))
            out = out.squeeze(0).cpu()[:, :self.chunk_size]

            if fade is not None and segments:
                # Crossfade the tail of the previous segment with the head of this one.
                segments[-1] = segments[-1].clone()
                segments[-1][:, -self.overlap:] = (
                    segments[-1][:, -self.overlap:] * (1 - fade)
                    + out[:, :self.overlap] * fade
                )
                out = out[:, self.overlap:]

            segments.append(out)
            pos_44100 += step
            chunk_idx += 1

            if verbose and chunk_idx % 10 == 0:
                print(f"  Chunk {chunk_idx}: {min(100, int(pos_44100 / chunks_44100_end * 100))}%")

        output = torch.cat(segments, dim=1)

        # Match the resampled input length.
        if output.shape[1] >= chunks_44100_end:
            output = output[:, :chunks_44100_end]
        else:
            output = F.pad(output, (0, chunks_44100_end - output.shape[1]))

        if resampler_out is not None:
            output = resampler_out(output)
            target_len = int(chunks_44100_end * target_sr / 44100 + 0.5)
            if output.shape[1] > target_len:
                output = output[:, :target_len]

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        torchaudio.save(output_path, output.cpu(), target_sr)

        if verbose:
            print(f"Saved: {output_path} ({output.shape[1] / target_sr:.1f} sec)")

        return output_path

    @torch.no_grad()
    def _model_forward(self, chunk):
        """Run SRNetwork + SpectralUNet on a chunk. Guarantees stereo."""
        # Ensure stereo
        if chunk.dim() == 3:
            if chunk.shape[1] == 1:
                chunk = chunk.repeat(1, 2, 1)
            elif chunk.shape[1] > 2:
                chunk = chunk[:, :2, :]

        pred = self.model(chunk)

        if self.spectral_unet is None:
            return pred

        # Spectral refinement
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
        refined_log_mag = self.spectral_unet(log_mag)
        refined_mag = torch.exp(refined_log_mag.squeeze(1))

        complex_spec = refined_mag * torch.exp(1j * phase.to(refined_mag.device))
        window2 = torch.hann_window(win, device="cpu", dtype=refined_mag.dtype)
        refined_wave = torch.istft(complex_spec, n_fft, hop, win, window=window2,
                                    length=pred.shape[-1])

        # Match channels
        if refined_wave.dim() == 2:
            refined_wave = refined_wave.unsqueeze(0)
        if refined_wave.shape[0] == 1 and pred.shape[0] == 1 and pred.dim() == 3:
            refined_wave = refined_wave.expand(-1, pred.shape[1], -1)

        # Blend
        final = 0.8 * refined_wave.to(pred.device) + 0.2 * pred
        return final