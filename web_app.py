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

"""Gradio Web UI for Audio Upscaler.

Features:
- Single file upscale (44.1kHz -> 96/192/384 kHz)
- QualityController (Judge-Jury-Executioner)
- Post-processing: TTA, Transient Restoration
- Batch processing
- Model info
"""
import os
import sys
import time

import gradio as gr

sys.path.insert(0, '.')

from inference import get_device, load_model, upscale_audio

# Global state
current_model = None
current_spectral_unet = None
current_config = None
current_discriminator = None
device = None


def init_device():
    global device
    device = get_device()
    return str(device)


def load_model_ui(checkpoint_path):
    """Load model from checkpoint."""
    global current_model, current_spectral_unet, current_config, current_discriminator
    if not checkpoint_path:
        return "No checkpoint selected"
    try:
        (current_model, current_spectral_unet,
         current_config, current_discriminator) = load_model(checkpoint_path, device)
        params = sum(p.numel() for p in current_model.parameters())
        spec_params = sum(p.numel() for p in current_spectral_unet.parameters())
        return f"SRNetwork: {params:,} params\nSpectralUNet: {spec_params:,} params"
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


def upscale_single(input_file, target_rate, use_qc, n_candidates, min_agreement,
                    use_tta, transient_strength, progress=None):
    """Upscale a single audio file with optional post-processing."""
    progress = progress or gr.Progress()
    if current_model is None:
        return None, "Load a model first!"
    if input_file is None:
        return None, "No input file!"

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, f"{base}_{target_rate}hz.wav")

    progress(0, desc="Starting...")
    start = time.time()

    try:
        upscale_audio(
            current_model,
            input_file,
            output_path,
            target_sr=int(target_rate),
            device=device,
            spectral_unet=current_spectral_unet,
            config=current_config,
            discriminator=current_discriminator,
            use_qc=use_qc,
            n_candidates=int(n_candidates),
            min_agreement=min_agreement,
            use_tta=use_tta,
            transient_strength=transient_strength,
        )
        elapsed = time.time() - start
        features = []
        if use_qc:
            features.append(f"QC({int(n_candidates)} candidates)")
        if use_tta:
            features.append("TTA")
        if transient_strength > 0:
            features.append(f"Transient({transient_strength})")
        feat_str = f" [{', '.join(features)}]" if features else ""
        return output_path, f"Done in {elapsed:.1f} sec{feat_str}"
    except Exception as e:  # noqa: BLE001
        return None, f"Error: {e}"


def get_model_info():
    """Get current model info."""
    if current_model is None:
        return "No model loaded"
    params = sum(p.numel() for p in current_model.parameters())
    spec_params = sum(p.numel() for p in current_spectral_unet.parameters()) if current_spectral_unet else 0
    return (
        f"SRNetwork: {params:,} params ({params * 4 / 1024**2:.1f} MB FP32)\n"
        f"SpectralUNet: {spec_params:,} params ({spec_params * 4 / 1024**2:.1f} MB FP32)\n"
        f"Total: {params + spec_params:,} params ({(params + spec_params) * 4 / 1024**2:.1f} MB FP32)"
    )


# Build UI
with gr.Blocks(title="Audio Upscaler", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Audio Upscaler")
    gr.Markdown("AI-powered audio super-resolution with hybrid waveform + spectral architecture.")

    with gr.Tabs():
        # Tab 1: Single File
        with gr.Tab("Upscale"):
            with gr.Row():
                with gr.Column():
                    input_file = gr.Audio(label="Input Audio", type="filepath")
                    target_rate = gr.Radio(
                        [96000, 192000, 384000],
                        value=96000,
                        label="Target Sample Rate"
                    )

                    gr.Markdown("### Quality Controller")
                    use_qc = gr.Checkbox(label="Enable QC (Judge-Jury-Executioner)", value=False)
                    n_candidates = gr.Slider(2, 10, value=3, step=1, label="Number of Candidates")
                    min_agreement = gr.Slider(0.3, 1.0, value=0.7, step=0.05, label="Min Agreement Threshold")

                    gr.Markdown("### Post-Processing")
                    use_tta = gr.Checkbox(label="TTA (phase inversion averaging)", value=False)
                    transient_strength = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Transient Restoration Strength")

                    upscale_btn = gr.Button("Upscale", variant="primary")

                with gr.Column():
                    output_file = gr.Audio(label="Output Audio", type="filepath")
                    status_text = gr.Textbox(label="Status", interactive=False)

            upscale_btn.click(
                upscale_single,
                inputs=[input_file, target_rate, use_qc, n_candidates, min_agreement,
                        use_tta, transient_strength],
                outputs=[output_file, status_text],
            )

        # Tab 2: Model
        with gr.Tab("Model"):
            checkpoint_path = gr.Textbox(
                label="Checkpoint Path",
                value="checkpoints/best_model.pt"
            )
            load_btn = gr.Button("Load Model")
            model_info = gr.Textbox(label="Model Info", interactive=False, lines=3)

            load_btn.click(load_model_ui, inputs=[checkpoint_path], outputs=[model_info])

            gr.Markdown("""
            ### Architecture
            - **SRNetwork**: 1D U-Net with GatedResBlocks (WaveNet-style), NoiseInjection, global residual
            - **SpectralUNet**: 2D U-Net on magnitude spectrograms for spectral refinement
            - **Discriminators**: MultiScale + MultiPeriod + MultiResolution (spectral)
            - **QualityController**: Judge-Jury-Executioner — N candidates, discriminator consensus
            - **Post-Processing**: TTA, Transient Restoration

            ### Training
            ```bash
            python train.py --data-dir D:\\path\\to\\library --config configs/default.yaml
            ```

            ### CLI Inference
            ```bash
            python inference.py input.wav -o output.wav -r 96000
            python inference.py input.wav --qc --n-candidates 5 --tta
            python inference.py input.wav --transient-strength 0.5
            ```
            """)

        # Tab 3: Batch
        with gr.Tab("Batch"):
            gr.Markdown("Batch processing coming soon — use CLI for now:")
            gr.Code("""
python inference.py input.wav -o output.wav -r 96000
python inference.py input.flac -o output.flac -r 192000 --qc --tta
            """, language="bash")

    # Footer
    gr.Markdown(f"Device: {init_device()}")


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)
