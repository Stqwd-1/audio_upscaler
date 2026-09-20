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

import numpy as np
import soundfile as sf
import os
from multiprocessing import Pool
from functools import partial

def create_file(freq, output_dir, sr=44100, duration=2):
    filepath = os.path.join(output_dir, f"freq_{freq:05d}hz.wav")
    if os.path.exists(filepath):
        return
    t = np.linspace(0, duration, sr * duration, dtype=np.float32)
    audio = 0.4 * np.sin(2 * np.pi * freq * t)
    harmonics = [2, 3, 0.5, 1.5]
    for i, h in enumerate(harmonics):
        amp = 0.15 / (i + 1)
        audio += amp * np.sin(2 * np.pi * freq * h * t)
    audio += np.random.randn(len(audio)).astype(np.float32) * 0.02
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
    sf.write(filepath, audio, sr)

if __name__ == "__main__":
    output_dir = "training_data/sine_waves"
    os.makedirs(output_dir, exist_ok=True)
    frequencies = list(range(1, 40001))
    print(f"Creating {len(frequencies)} audio files...")
    with Pool(8) as pool:
        pool.map(partial(create_file, output_dir=output_dir), frequencies)
    print(f"Done! Files created in {output_dir}")
