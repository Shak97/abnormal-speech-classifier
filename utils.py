from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment


def load_audio_any(path: str):
    path = str(path)


    try:
        wav, sr = sf.read(path, always_2d=False)

        # wav can be float or int numpy array
        if np.issubdtype(wav.dtype, np.integer):
            max_val = np.iinfo(wav.dtype).max
            wav = wav.astype(np.float32) / max_val
        else:
            wav = wav.astype(np.float32)

        wav = torch.from_numpy(wav)
        return wav, sr

    except Exception:
        pass


    audio = AudioSegment.from_file(path)

    samples = np.array(audio.get_array_of_samples())


    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels)  # [T, C]

    scale = float(1 << (8 * audio.sample_width - 1))
    wav = samples.astype(np.float32) / scale

    wav = torch.from_numpy(wav)
    sr = audio.frame_rate
    return wav, sr