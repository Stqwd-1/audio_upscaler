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

"""Export SRNetwork to ONNX format for hardware-accelerated inference.

Usage:
    python export_onnx.py --checkpoint checkpoints/best_model.pt --output models/sr_network.onnx
    python export_onnx.py --checkpoint checkpoints/best_model.pt --output models/sr_network.onnx --dynamic-axes
"""
import argparse
import sys

import torch

sys.path.insert(0, '.')

from models.spectral_unet import SpectralUNet
from models.sr_network import SRNetwork


def load_model(checkpoint_path: str):
    """Load generator + SpectralUNet from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = ckpt['config']

    sr_cfg = config['model']['sr_network']
    generator = SRNetwork(**sr_cfg)
    generator.load_state_dict(ckpt['generator'], strict=False)
    generator.eval()

    spectral_unet = SpectralUNet(in_channels=1, base_channels=32)
    if 'spectral_unet' in ckpt:
        spectral_unet.load_state_dict(ckpt['spectral_unet'], strict=False)
    spectral_unet.eval()

    return generator, spectral_unet, config


def export_to_onnx(
    generator,
    output_path: str,
    cond_dim: int = 0,
    dynamic_axes: bool = False,
    opset_version: int = 17,
):
    """Export SRNetwork to ONNX.

    Args:
        generator: Trained SRNetwork model
        output_path: Output ONNX file path
        cond_dim: FiLM conditioning dimension (0 = no conditioning)
        dynamic_axes: Enable dynamic input/output axes
        opset_version: ONNX opset version
    """
    # Dummy input
    batch_size = 1
    channels = 2
    samples = 44100
    dummy_input = torch.randn(batch_size, channels, samples)

    # Conditioning vector
    dummy_cond = None
    if cond_dim > 0:
        dummy_cond = torch.full((batch_size, 1), 0.36)

    # Export
    print(f"Exporting to ONNX: {output_path}")
    print(f"  Input shape: {dummy_input.shape}")
    if dummy_cond is not None:
        print(f"  Cond shape: {dummy_cond.shape}")

    input_names = ['waveform']
    output_names = ['output']
    dynamic_axes_dict = None

    if dynamic_axes:
        dynamic_axes_dict = {
            'waveform': {0: 'batch', 2: 'time'},
            'output': {0: 'batch', 2: 'time'},
        }
        if dummy_cond is not None:
            input_names.append('conditioning')
            dynamic_axes_dict['conditioning'] = {0: 'batch'}

    args = (dummy_input,)
    if dummy_cond is not None:
        args = (dummy_input, dummy_cond)

    torch.onnx.export(
        generator,
        args,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
    )

    # Verify
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported: {output_path} ({size_mb:.1f} MB)")

    # Quick inference test
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        ort_inputs = {'waveform': dummy_input.numpy()}
        if dummy_cond is not None:
            ort_inputs['conditioning'] = dummy_cond.numpy()
        ort_output = session.run(None, ort_inputs)
        print(f"ONNX Runtime verification OK: output shape {ort_output[0].shape}")
    except ImportError:
        print("ONNX Runtime not installed — skipping verification")
    except Exception as e:  # noqa: BLE001
        print(f"ONNX Runtime verification failed: {e}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SRNetwork to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--output", default="models/sr_network.onnx", help="Output ONNX path")
    parser.add_argument("--dynamic-axes", action="store_true", help="Enable dynamic input shapes")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    generator, spectral_unet, config = load_model(args.checkpoint)
    cond_dim = config['model']['sr_network'].get('cond_dim', 0)

    params = sum(p.numel() for p in generator.parameters())
    print(f"SRNetwork: {params:,} params")
    print(f"FiLM cond_dim: {cond_dim}")

    export_to_onnx(
        generator,
        args.output,
        cond_dim=cond_dim,
        dynamic_axes=args.dynamic_axes,
        opset_version=args.opset,
    )
