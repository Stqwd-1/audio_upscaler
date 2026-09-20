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

"""Hybrid Quality Control — Judge-Jury-Executioner system.

Three-stage quality pipeline to eliminate hallucinations:

1. JUDGE: Each discriminator independently scores every candidate
          using multi-resolution spectrogram analysis (0-1 realism probability).
2. JURY:  Candidates that fail consensus are rejected.
          Agreement check: do all judges agree?
          Variance check: is there disagreement?
3. EXECUTIONER: Only candidates above threshold survive.
                Best one is selected by highest mean score.
"""
import torch
import torch.nn as nn


class QualityController:
    """Judge-Jury-Executioner quality control.

    Uses MultiResolutionDiscriminator.score_candidate() for scoring.

    Usage:
        from models.discriminators import MultiResolutionDiscriminator
        from inference import _model_forward
        mrd = MultiResolutionDiscriminator()
        qc = QualityController(model, discriminators=[mrd], n_candidates=5,
                               forward_fn=_model_forward, spectral_unet=unet, cond=cond)
        best_audio = qc.upscale(low_audio)
    """
    def __init__(
        self,
        generator,
        discriminators: list,
        n_candidates: int = 5,
        threshold: float = 0.3,
        min_agreement: float = 0.7,
        forward_fn=None,
        spectral_unet=None,
        cond=None,
    ):
        self.generator = generator
        self.discriminators = nn.ModuleList(discriminators)
        self.n_candidates = n_candidates
        self.threshold = threshold
        self.min_agreement = min_agreement
        self.forward_fn = forward_fn
        self.spectral_unet = spectral_unet
        self.cond = cond

    @torch.no_grad()
    def generate_candidates(self, low_audio: torch.Tensor) -> list[torch.Tensor]:
        """Generate N candidates with different noise seeds."""
        candidates = []
        was_training = self.generator.training
        self.generator.train()  # Keep noise injection active

        for _ in range(self.n_candidates):
            if self.forward_fn is not None:
                # Same path as regular inference (incl. spectral refinement)
                candidate = self.forward_fn(self.generator, low_audio, self.spectral_unet, self.cond)
            elif self.cond is not None:
                candidate = self.generator(low_audio, self.cond)
            else:
                candidate = self.generator(low_audio)
            candidates.append(candidate)

        if not was_training:
            self.generator.eval()

        return candidates

    @torch.no_grad()
    def judge(self, candidate: torch.Tensor) -> dict:
        """JUDGE stage: Score candidate with each discriminator.

        Uses score_candidate() for multi-resolution spectrogram scoring.
        Returns scores from every judge (discriminator).
        """
        scores = []
        for disc in self.discriminators:
            disc.eval()
            if hasattr(disc, 'score_candidate'):
                # MultiResolutionDiscriminator — uses spectrogram scoring
                score = disc.score_candidate(candidate)
                scores.append(score)
            else:
                # Fallback for other discriminators
                outputs = disc(candidate, candidate)
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    score_list = outputs[0]
                    if isinstance(score_list, list):
                        for s in score_list:
                            scores.append(torch.sigmoid(s).mean().item())
                    else:
                        scores.append(torch.sigmoid(score_list).mean().item())

        return {
            "scores": scores,
            "mean": sum(scores) / max(len(scores), 1),
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
        }

    @torch.no_grad()
    def jury(self, all_judgments: list[dict]) -> list[dict]:
        """JURY stage: Aggregate scores, reject candidates that fail consensus."""
        results = []
        for i, judgment in enumerate(all_judgments):
            scores = judgment["scores"]

            if not scores:
                results.append({"pass": False, "reason": "no scores", "idx": i, "mean_score": 0})
                continue

            # Count how many judges agree (score above threshold)
            agree_count = sum(1 for s in scores if s > self.threshold)
            agreement = agree_count / len(scores)

            # Compute variance
            mean_score = judgment["mean"]
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores) if len(scores) > 1 else 0

            # Pass criteria
            passes = (
                agreement >= self.min_agreement and
                variance < 0.1 and
                judgment["mean"] > self.threshold
            )

            results.append({
                "pass": passes,
                "idx": i,
                "agreement": agreement,
                "variance": variance,
                "mean_score": judgment["mean"],
                "reason": "OK" if passes else f"agreement={agreement:.2f}, var={variance:.4f}",
            })

        return results

    @torch.no_grad()
    def executioner(self, candidates: list[torch.Tensor], jury_results: list[dict]) -> torch.Tensor:
        """EXECUTIONER stage: Eliminate rejected, select best."""
        survivors = [(r["mean_score"], r["idx"]) for r in jury_results if r["pass"]]

        if not survivors:
            # Fallback: best overall
            fallback = max(jury_results, key=lambda r: r.get("mean_score", 0))
            return candidates[fallback["idx"]]

        survivors.sort(key=lambda x: x[0], reverse=True)
        return candidates[survivors[0][1]]

    @torch.no_grad()
    def upscale(self, low_audio: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """Full Judge-Jury-Executioner pipeline."""
        candidates = self.generate_candidates(low_audio)
        judgments = [self.judge(c) for c in candidates]
        jury_results = self.jury(judgments)

        if verbose:
            for i, (j, r) in enumerate(zip(judgments, jury_results)):
                status = "PASS" if r["pass"] else "FAIL"
                print(f"  Candidate {i}: {status} | score={j['mean']:.3f} | {r['reason']}")

        best = self.executioner(candidates, jury_results)
        return best
