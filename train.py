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

import argparse
import random
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

from data.dataset import create_dataloaders
from data.degradation import AdvancedDegradation
from losses import (
    AdversarialLoss,
    FeatureMatchingLoss,
    HFExciterLoss,
    HighFrequencyLoss,
    MelLoss,
    SpectralLoss,
    WaveLoss,
)
from models.discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiScaleDiscriminator,
)
from models.hifi_gan import Generator as HiFiGANGenerator
from models.spectral_unet import SpectralUNet
from models.sr_network import SRNetwork
from utils.metrics import calculate_lsd, calculate_psnr, calculate_si_sdr


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif HAS_DIRECTML:
        return torch_directml.device()
    return torch.device("cpu")


def _stft_device(tensor_device):
    """Pick STFT device: CPU for DirectML (ComplexFloat unsupported), tensor device otherwise."""
    if HAS_DIRECTML and str(tensor_device).startswith("privateuseone"):
        return torch.device("cpu")
    return tensor_device


def stft_mag(x, n_fft=1024, hop=256, win=1024):
    """Compute magnitude spectrogram. Uses CPU on DirectML, tensor device on CUDA/CPU."""
    orig_device = x.device
    dev = _stft_device(orig_device)
    if x.dim() == 3:
        b, c, t = x.shape
        x = x.reshape(b * c, t)
    x = x.to(dev)
    window = torch.hann_window(win, device=dev, dtype=x.dtype)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                       window=window, return_complex=True)
    return torch.abs(spec).to(orig_device)


def istft_mag(mag, phase, n_fft=1024, hop=256, win=1024, length=None):
    """Reconstruct waveform from magnitude + phase. Uses CPU on DirectML."""
    orig_device = mag.device
    dev = _stft_device(orig_device)
    complex_spec = mag.to(dev) * torch.exp(1j * phase.to(dev))
    window = torch.hann_window(win, device=dev, dtype=mag.dtype)
    waveform = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop, win_length=win,
                            window=window, length=length)
    return waveform.to(orig_device)



def toggle_grad(model, requires_grad):
    if model is not None:
        for p in model.parameters():
            p.requires_grad = requires_grad

def load_chkpt(model, state, name):
    if model is None: return
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys: print(f"Warning ({name}): missing keys {result.missing_keys}")
    if result.unexpected_keys: print(f"Warning ({name}): unexpected keys {result.unexpected_keys}")

def train(config, data_dir: str, output_dir: str, resume_path: str | None = None):
    if isinstance(config, str):
        config = load_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # --- Models ---
    generator_type = config["model"].get("generator", "sr_network")
    if generator_type == "hifi_gan":
        hifi_cfg = config["model"].get("hifi_gan", {})
        generator = HiFiGANGenerator(**hifi_cfg).to(device)
        print(f"HiFi-GAN Generator: {sum(p.numel() for p in generator.parameters()):,} params")
    else:
        sr_cfg = config["model"]["sr_network"]
        generator = SRNetwork(**sr_cfg).to(device)
        print(f"SRNetwork: {sr_cfg['in_channels']}-ch input, base_channels={sr_cfg['base_channels']}")
    spectral_unet = SpectralUNet(in_channels=1, base_channels=32).to(device)

    discriminator_s = MultiScaleDiscriminator().to(device)
    discriminator_p = MultiPeriodDiscriminator().to(device)
    discriminator_spec = MultiResolutionDiscriminator().to(device)

    gen_params = sum(p.numel() for p in generator.parameters())
    spec_params = sum(p.numel() for p in spectral_unet.parameters())
    disc_params = (
        sum(p.numel() for p in discriminator_s.parameters())
        + sum(p.numel() for p in discriminator_p.parameters())
        + sum(p.numel() for p in discriminator_spec.parameters())
    )
    print(f"SRNetwork: {gen_params:,} params ({gen_params * 4 / 1024**2:.1f} MB)")
    print(f"SpectralUNet: {spec_params:,} params ({spec_params * 4 / 1024**2:.1f} MB)")
    print(f"Discriminators: {disc_params:,} params ({disc_params * 4 / 1024**2:.1f} MB)")

    # --- Optimizers ---
    gen_params_list = list(generator.parameters()) + list(spectral_unet.parameters())
    disc_params_list = (
        list(discriminator_s.parameters())
        + list(discriminator_p.parameters())
        + list(discriminator_spec.parameters())
    )

    opt_g = torch.optim.AdamW(
        gen_params_list,
        lr=config["training"]["learning_rate"],
        betas=config["training"]["betas"],
        weight_decay=config["training"]["weight_decay"],
    )
    opt_d = torch.optim.AdamW(
        disc_params_list,
        lr=config["training"]["learning_rate"],
        betas=config["training"]["betas"],
        weight_decay=config["training"]["weight_decay"],
    )

    # --- LR Schedulers with warmup ---
    warmup_epochs = config["training"].get("warmup_epochs", 0)
    total_epochs = config["training"]["epochs"]

    warmup_g = torch.optim.lr_scheduler.LinearLR(
        opt_g, start_factor=0.01, total_iters=max(warmup_epochs, 1)
    )
    cosine_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=total_epochs - warmup_epochs
    )
    scheduler_g = torch.optim.lr_scheduler.SequentialLR(
        opt_g, schedulers=[warmup_g, cosine_g], milestones=[warmup_epochs]
    )

    warmup_d = torch.optim.lr_scheduler.LinearLR(
        opt_d, start_factor=0.01, total_iters=max(warmup_epochs, 1)
    )
    cosine_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_d, T_max=total_epochs - warmup_epochs
    )
    scheduler_d = torch.optim.lr_scheduler.SequentialLR(
        opt_d, schedulers=[warmup_d, cosine_d], milestones=[warmup_epochs]
    )

    # --- Losses ---
    wave_loss = WaveLoss(
        loss_weight=config["losses"].get("wave", {}).get("loss_weight", 1.0),
    ).to(device)
    spectral_loss = SpectralLoss(
        fft_sizes=config["losses"]["spectral"]["fft_sizes"],
        hop_sizes=config["losses"]["spectral"]["hop_sizes"],
        win_sizes=config["losses"]["spectral"]["win_sizes"],
        loss_weight=config["losses"]["spectral"]["loss_weight"],
    ).to(device)
    mel_loss = MelLoss(
        n_mels=config["losses"]["mel"]["n_mels"],
        loss_weight=config["losses"]["mel"]["loss_weight"],
    ).to(device)
    adv_loss = AdversarialLoss(loss_weight=config["losses"]["adversarial"]["loss_weight"])
    fm_loss = FeatureMatchingLoss(loss_weight=config["losses"]["feature_matching"]["loss_weight"])

    # HF Exciter Loss (optional)
    hf_cfg = config["losses"].get("hf_exciter", {})
    use_hf_exciter = hf_cfg.get("enabled", False)
    if use_hf_exciter:
        hf_exciter_loss = HFExciterLoss(
            sample_rate=config["data"]["sample_rate"],
            hf_start_freq=hf_cfg.get("hf_start_freq", 16000),
            n_fft=hf_cfg.get("n_fft", 4096),
            loss_weight=hf_cfg.get("loss_weight", 2.0),
        ).to(device)
        print(f"HF Exciter Loss enabled (start={hf_cfg.get('hf_start_freq', 16000)} Hz)")

    # High-Frequency Suppression Loss (penalize white noise in upper spectrum)
    hf_sup_cfg = config["losses"].get("high_frequency", {})
    hf_cutoff = hf_sup_cfg.get("cutoff_freq", 15000)
    hf_sup_loss = HighFrequencyLoss(
        n_fft=2048,
        sample_rate=config["data"].get("sample_rate", 44100),
        cutoff_freq=hf_cutoff,
        loss_weight=hf_sup_cfg.get("loss_weight", 5.0),
    ).to(device)
    print(f"High-Frequency Suppression Loss enabled (cutoff={hf_cutoff} Hz, weight={hf_sup_cfg.get('loss_weight', 5.0)})")

    # --- Degradation pipeline ---
    degradation = AdvancedDegradation(sample_rate=config["data"]["sample_rate"]).to(device)

    # STFT params for spectral branch
    spec_n_fft = 1024
    spec_hop = 256
    spec_win = 1024

    # --- AMP (mixed precision) ---
    use_amp = config["training"].get("mixed_precision", False) and device.type == "cuda"
    scaler_g = torch.amp.GradScaler("cuda") if use_amp else None
    scaler_d = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("AMP enabled (mixed precision)")

    # --- Gradient accumulation ---
    accum_steps = config["training"].get("gradient_accumulation", 1)
    grad_clip_max_norm = config["training"].get("grad_clip_max_norm", 0.5)

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    if resume_path:
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        load_chkpt(generator, checkpoint["generator"], "generator")
        if "spectral_unet" in checkpoint:
            load_chkpt(spectral_unet, checkpoint["spectral_unet"], "spectral_unet")
        load_chkpt(discriminator_s, checkpoint["discriminator_s"], "discriminator_s")
        load_chkpt(discriminator_p, checkpoint["discriminator_p"], "discriminator_p")
        if "discriminator_spec" in checkpoint:
            load_chkpt(discriminator_spec, checkpoint["discriminator_spec"], "discriminator_spec")
        opt_g.load_state_dict(checkpoint["opt_g"])
        opt_d.load_state_dict(checkpoint["opt_d"])
        if use_amp and "scaler_g" in checkpoint:
            scaler_g.load_state_dict(checkpoint["scaler_g"])
        if use_amp and "scaler_d" in checkpoint:
            scaler_d.load_state_dict(checkpoint["scaler_d"])
        start_epoch = checkpoint["epoch"] + 1
        if "best_val_loss" in checkpoint:
            best_val_loss_resume = checkpoint["best_val_loss"]
        if "patience_counter" in checkpoint:
            patience_counter_resume = checkpoint["patience_counter"]
        for _ in range(start_epoch):
            scheduler_g.step()
            scheduler_d.step()
        print(f"Resumed from epoch {start_epoch}")

    # --- Dataset ---
    val_split = config["data"].get("val_split", 0.0)
    train_loader, val_loader = create_dataloaders(config, data_dir)
    print(f"Train: {len(train_loader.dataset)} files, Val: {len(val_loader.dataset) if val_loader else 0} files")

    writer = SummaryWriter(output_dir / "logs")
    num_sub_disc = 11  # 3 MSD + 5 MPD + 3 MRD

    # --- Early Stopping ---
    best_val_loss = best_val_loss_resume if "best_val_loss_resume" in locals() else float("inf")
    patience = config["training"].get("early_stopping_patience", 0)
    patience_counter = patience_counter_resume if "patience_counter_resume" in locals() else 0

    for epoch in range(start_epoch, config["training"]["epochs"]):
        generator.train()
        spectral_unet.train()
        discriminator_s.train()
        discriminator_p.train()
        discriminator_spec.train()

        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        l_hf = torch.tensor(0.0)  # Initialize for logging
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}")

        for batch in pbar:
            low_audio = batch["low_audio"].to(device)
            high_audio = batch["audio"].to(device)

            # --- Degradation on low-res input ---
            degraded = degradation(low_audio)

            # --- Conditioning vector (normalized sample rate) ---
            cond = None
            if hasattr(generator, 'cond_encoder') and generator.cond_encoder is not None:
                # Conditioning: normalized low_sample_rate / target_sample_rate
                low_sr = config["data"].get("low_sample_rate", 16000)
                target_sr = config["data"].get("sample_rate", 44100)
                cond_val = low_sr / target_sr  # e.g., 16000/44100 ≈ 0.36
                cond = torch.full((degraded.shape[0], 1), cond_val, device=device, dtype=degraded.dtype)

            # === GENERATOR STEP ===
            if use_amp:
                amp_ctx = torch.amp.autocast(device_type="cuda")
            else:
                amp_ctx = torch.autocast("cpu", enabled=False)

            with amp_ctx:
                # Step 1: SRNetwork waveform prediction
                pred_waveform = generator(degraded, cond) if cond is not None else generator(degraded)

                # Trim to common length
                min_len = min(pred_waveform.shape[-1], high_audio.shape[-1])
                pred_waveform = pred_waveform[..., :min_len]
                high_crop = high_audio[..., :min_len]

                # Step 2: Spectral refinement
                # STFT of predicted and target (on CPU for DirectML compatibility)
                pred_spec = stft_mag(pred_waveform, spec_n_fft, spec_hop, spec_win)
                target_spec = stft_mag(high_crop, spec_n_fft, spec_hop, spec_win)

                # Log magnitude for SpectralUNet input
                pred_log_mag = torch.log(pred_spec + 1e-7).unsqueeze(1)  # (B, 1, F, T)
                target_log_mag = torch.log(target_spec + 1e-7).unsqueeze(1)

                refined_log_mag = spectral_unet(pred_log_mag)
                refined_mag = torch.exp(refined_log_mag.squeeze(1))  # (B, F, T)

                # Phase from predicted waveform
                x_phase = pred_waveform.mean(dim=1) if pred_waveform.dim() == 3 else pred_waveform
                phase_dev = _stft_device(pred_waveform.device)
                x_phase_dev = x_phase.to(phase_dev)
                window = torch.hann_window(spec_win, device=phase_dev, dtype=x_phase_dev.dtype)
                phase = torch.angle(
                    torch.stft(x_phase_dev, spec_n_fft, spec_hop, spec_win,
                               window=window, return_complex=True)
                ).to(pred_waveform.device)

                # ISTFT with refined magnitude + original phase
                reconstructed = istft_mag(refined_mag, phase, spec_n_fft, spec_hop, spec_win,
                                           length=min_len)

                reconstructed = reconstructed.view(pred_waveform.shape[0], -1, reconstructed.shape[-1])
                if reconstructed.shape[1] == 1 and pred_waveform.dim() == 3 and pred_waveform.shape[1] > 1:
                    reconstructed = reconstructed.expand(-1, pred_waveform.shape[1], -1)

                final = reconstructed.to(device)

                # --- Generator losses ---
                toggle_grad(discriminator_s, False)
                toggle_grad(discriminator_p, False)
                toggle_grad(discriminator_spec, False)
                # Waveform losses
                l_wave = wave_loss(final, high_crop)

                # Spectral loss (on CPU for DirectML)
                l_spectral = spectral_loss(final, high_crop)

                # Mel loss (on CPU for DirectML)
                l_mel = mel_loss(final, high_crop)

                # Adversarial + feature matching — waveform discriminators (mono for MSD/MPD)
                high_mono_g = high_crop.mean(dim=1, keepdim=True) if high_crop.dim() == 3 and high_crop.shape[1] > 1 else high_crop
                final_mono_g = final.mean(dim=1, keepdim=True) if final.dim() == 3 and final.shape[1] > 1 else final
                d_s_out = discriminator_s(high_mono_g, final_mono_g)
                d_p_out = discriminator_p(high_mono_g, final_mono_g)

                # unpack: (y_d_rs, y_d_gs, fmap_rs, fmap_gs)
                y_d_rs_s, y_d_gs_s, fmap_rs_s, fmap_gs_s = d_s_out
                y_d_rs_p, y_d_gs_p, fmap_rs_p, fmap_gs_p = d_p_out

                l_adv_s = adv_loss(y_d_gs_s, is_real=False)
                l_adv_p = adv_loss(y_d_gs_p, is_real=False)
                l_fm_s = fm_loss(fmap_rs_s, fmap_gs_s)
                l_fm_p = fm_loss(fmap_rs_p, fmap_gs_p)

                # Spectral discriminator adversarial
                d_spec_out = discriminator_spec(high_crop, final)
                y_d_rs_spec, y_d_gs_spec, fmap_rs_spec, fmap_gs_spec = d_spec_out
                l_adv_spec = adv_loss(y_d_gs_spec, is_real=False)
                l_fm_spec = fm_loss(fmap_rs_spec, fmap_gs_spec)

                l_adv = l_adv_s + l_adv_p + l_adv_spec
                l_fm = l_fm_s + l_fm_p + l_fm_spec

                loss_g = (l_wave + l_spectral + l_mel + l_adv + l_fm) / accum_steps

                # HF Exciter Loss (optional)
                if use_hf_exciter:
                    l_hf = hf_exciter_loss(final, high_crop)
                    loss_g = loss_g + l_hf / accum_steps

                # High-Frequency Suppression Loss (penalize white noise cheating)
                l_hf_sup = hf_sup_loss(final, high_crop)
                loss_g = loss_g + l_hf_sup / accum_steps

            if use_amp:
                scaler_g.scale(loss_g).backward()
            else:
                loss_g.backward()

            if (global_step + 1) % accum_steps == 0:
                if use_amp:
                    scaler_g.unscale_(opt_g)
                    torch.nn.utils.clip_grad_norm_(gen_params_list, grad_clip_max_norm)
                    scaler_g.step(opt_g)
                    scaler_g.update()
                else:
                    torch.nn.utils.clip_grad_norm_(gen_params_list, grad_clip_max_norm)
                    opt_g.step()
                opt_g.zero_grad()

            # === DISCRIMINATOR STEP ===
            toggle_grad(discriminator_s, True)
            toggle_grad(discriminator_p, True)
            toggle_grad(discriminator_spec, True)
            with torch.no_grad():
                final_det = final.detach()

            with amp_ctx:
                # Convert to mono for waveform discriminators
                high_mono = high_crop.mean(dim=1, keepdim=True) if high_crop.dim() == 3 and high_crop.shape[1] > 1 else high_crop
                final_mono = final.mean(dim=1, keepdim=True) if final.dim() == 3 and final.shape[1] > 1 else final
                final_det_mono = final_det.mean(dim=1, keepdim=True) if final_det.dim() == 3 and final_det.shape[1] > 1 else final_det

                # MSD — real vs real, real vs fake
                d_s_real = discriminator_s(high_mono, high_mono)
                d_s_fake = discriminator_s(high_mono, final_det_mono)
                l_d_s = adv_loss(d_s_fake[1], is_real=False) + adv_loss(d_s_real[0], is_real=True)

                # MPD — real vs real, real vs fake
                d_p_real = discriminator_p(high_mono, high_mono)
                d_p_fake = discriminator_p(high_mono, final_det_mono)
                l_d_p = adv_loss(d_p_fake[1], is_real=False) + adv_loss(d_p_real[0], is_real=True)

                # Spectral disc — real vs real, real vs fake (uses spectrograms, needs mono)
                d_spec_real = discriminator_spec(high_crop, high_crop)
                d_spec_fake = discriminator_spec(high_crop, final_det)
                l_d_spec = adv_loss(d_spec_fake[1], is_real=False) + adv_loss(d_spec_real[0], is_real=True)

                loss_d = (l_d_s + l_d_p + l_d_spec) / num_sub_disc / accum_steps

            if use_amp:
                scaler_d.scale(loss_d).backward()
            else:
                loss_d.backward()

            if (global_step + 1) % accum_steps == 0:
                if use_amp:
                    scaler_d.unscale_(opt_d)
                    torch.nn.utils.clip_grad_norm_(disc_params_list, grad_clip_max_norm)
                    scaler_d.step(opt_d)
                    scaler_d.update()
                else:
                    torch.nn.utils.clip_grad_norm_(disc_params_list, grad_clip_max_norm)
                    opt_d.step()
                opt_d.zero_grad()

            # --- Logging ---
            epoch_loss_g += loss_g.item() * accum_steps
            epoch_loss_d += loss_d.item() * accum_steps
            pbar.set_postfix({"loss_g": f"{loss_g.item() * accum_steps:.4f}", "loss_d": f"{loss_d.item() * accum_steps:.4f}"})
            writer.add_scalars("loss", {"generator": loss_g.item(), "discriminator": loss_d.item()}, global_step)
            writer.add_scalars("loss_detail", {
                "wave": l_wave.item(),
                "spectral": l_spectral.item(),
                "mel": l_mel.item(),
                "adv": l_adv.item(),
                "fm": l_fm.item(),
                "hf_exciter": l_hf.item() if use_hf_exciter else 0.0,
                "hf_suppression": l_hf_sup.item(),
            }, global_step)
            global_step += 1

        # LR schedulers are epoch-based (warmup + cosine), so they step
        # once per epoch — not per batch. Resuming replays these steps.
        
        if global_step % accum_steps != 0:
            if use_amp:
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(gen_params_list, grad_clip_max_norm)
                scaler_g.step(opt_g)
                scaler_g.update()
                
                scaler_d.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(disc_params_list, grad_clip_max_norm)
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                torch.nn.utils.clip_grad_norm_(gen_params_list, grad_clip_max_norm)
                opt_g.step()
                torch.nn.utils.clip_grad_norm_(disc_params_list, grad_clip_max_norm)
                opt_d.step()
            opt_g.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)

        scheduler_g.step()
        scheduler_d.step()

        avg_g = epoch_loss_g / len(train_loader)
        avg_d = epoch_loss_d / len(train_loader)
        writer.add_scalars("epoch_loss", {"generator": avg_g, "discriminator": avg_d}, epoch)
        print(f"Epoch {epoch + 1} - train_loss_g: {avg_g:.4f}, train_loss_d: {avg_d:.4f}")

        # --- Validation ---
        if val_loader is not None:
            generator.eval()
            spectral_unet.eval()
            val_loss_g = 0.0
            val_psnr = 0.0
            val_si_sdr = 0.0
            val_lsd = 0.0
            val_count = 0

            with torch.no_grad():
                # Keep validation degradation fixed so metrics are comparable across epochs
                torch.manual_seed(0)
                random.seed(0)
                for batch in val_loader:
                    low_audio = batch["low_audio"].to(device)
                    high_audio = batch["audio"].to(device)
                    degraded = degradation(low_audio)

                    # Conditioning for validation
                    cond = None
                    if hasattr(generator, 'cond_encoder') and generator.cond_encoder is not None:
                        low_sr = config["data"].get("low_sample_rate", 16000)
                        target_sr = config["data"].get("sample_rate", 44100)
                        cond_val = low_sr / target_sr
                        cond = torch.full((degraded.shape[0], 1), cond_val, device=device, dtype=degraded.dtype)

                    pred_waveform = generator(degraded, cond) if cond is not None else generator(degraded)

                    min_len = min(pred_waveform.shape[-1], high_audio.shape[-1])
                    pred_waveform = pred_waveform[..., :min_len]
                    high_crop = high_audio[..., :min_len]

                    # Spectral refinement
                    pred_spec = stft_mag(pred_waveform, spec_n_fft, spec_hop, spec_win)
                    target_spec = stft_mag(high_crop, spec_n_fft, spec_hop, spec_win)
                    pred_log_mag = torch.log(pred_spec + 1e-7).unsqueeze(1)
                    refined_log_mag = spectral_unet(pred_log_mag)
                    refined_mag = torch.exp(refined_log_mag.squeeze(1))

                    x_phase = pred_waveform.mean(dim=1) if pred_waveform.dim() == 3 else pred_waveform
                    phase_dev = _stft_device(pred_waveform.device)
                    x_phase_dev = x_phase.to(phase_dev)
                    window = torch.hann_window(spec_win, device=phase_dev, dtype=x_phase_dev.dtype)
                    phase = torch.angle(
                        torch.stft(x_phase_dev, spec_n_fft, spec_hop, spec_win,
                                   window=window, return_complex=True)
                    ).to(pred_waveform.device)

                    reconstructed = istft_mag(refined_mag, phase, spec_n_fft, spec_hop, spec_win,
                                               length=min_len)
                    reconstructed = reconstructed.view(pred_waveform.shape[0], -1, reconstructed.shape[-1])
                    if reconstructed.shape[1] == 1 and pred_waveform.dim() == 3 and pred_waveform.shape[1] > 1:
                        reconstructed = reconstructed.expand(-1, pred_waveform.shape[1], -1)
                    final = reconstructed.to(device)

                    val_loss_g += wave_loss(final, high_crop).item()
                    val_psnr += calculate_psnr(final, high_crop)
                    val_si_sdr += calculate_si_sdr(final, high_crop)
                    final_spec = stft_mag(final, spec_n_fft, spec_hop, spec_win)
                    val_lsd += calculate_lsd(target_spec, final_spec)
                    val_count += 1

            val_loss_g /= max(val_count, 1)
            val_psnr /= max(val_count, 1)
            val_si_sdr /= max(val_count, 1)
            val_lsd /= max(val_count, 1)

            writer.add_scalars("val_loss", {"generator": val_loss_g}, epoch)
            writer.add_scalars("val_metrics", {"psnr": val_psnr, "si_sdr": val_si_sdr, "lsd": val_lsd}, epoch)
            print(f"  Val - loss: {val_loss_g:.4f}, PSNR: {val_psnr:.2f} dB, SI-SDR: {val_si_sdr:.2f} dB, LSD: {val_lsd:.4f}")

            # --- Early Stopping + Best Model ---
            if val_loss_g < best_val_loss:
                best_val_loss = val_loss_g
                patience_counter = 0
                best_path = output_dir / "best_model.pt"
                best_ckpt = {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "spectral_unet": spectral_unet.state_dict(),
                    "discriminator_s": discriminator_s.state_dict(),
                    "discriminator_p": discriminator_p.state_dict(),
                    "discriminator_spec": discriminator_spec.state_dict(),
                    "global_step": global_step,
                    "scheduler_g": scheduler_g.state_dict(),
                    "scheduler_d": scheduler_d.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "config": config,
                    "best_val_loss": best_val_loss,
                }
                if use_amp:
                    best_ckpt["scaler_g"] = scaler_g.state_dict()
                    best_ckpt["scaler_d"] = scaler_d.state_dict()
                torch.save(best_ckpt, best_path)
                print(f"  Best model saved: {best_path} (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
                    break

        if (epoch + 1) % config["training"]["save_every"] == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            ckpt = {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "spectral_unet": spectral_unet.state_dict(),
                "discriminator_s": discriminator_s.state_dict(),
                "discriminator_p": discriminator_p.state_dict(),
                "discriminator_spec": discriminator_spec.state_dict(),
                "global_step": global_step,
                "scheduler_g": scheduler_g.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "config": config,
            }
            if use_amp:
                ckpt["scaler_g"] = scaler_g.state_dict()
                ckpt["scaler_d"] = scaler_d.state_dict()
            torch.save(ckpt, ckpt_path)
            print(f"Saved: {ckpt_path}")

    print("Training complete!")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio upscaler (hybrid waveform + spectral)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", required=True, help="Folder with audio files (FLAC/WAV, nested OK)")
    parser.add_argument("--output-dir", default="checkpoints", help="Where to save checkpoints and logs")
    parser.add_argument("--resume", default=None, help="Resume from a checkpoint (.pt)")
    args = parser.parse_args()
    train(args.config, args.data_dir, args.output_dir, args.resume)
