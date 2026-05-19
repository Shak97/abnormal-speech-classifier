import random
from pathlib import Path
from typing import List, Dict, Optional
import os, sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
# from utils import load_audio_any # Assuming this helper is available
import torchaudio

torchaudio.set_audio_backend("soundfile")

class ControlSetDataset(Dataset):
    """
    Updated to use MFCC + Delta + Delta-Delta features to match the main dataset.
    This ensures the 'Normal' control class is processed identically to the 'Abnormal' class.
    """
    def __init__(
        self,
        data_root: str,
        sampling_rate: int = 16000,
        max_files_per_dir: int = 250,
        seed: int = 0,
        normalize_wav: bool = True,
        n_mfcc: int = 40,
        output_format: str = "cnn",
        target_frames: int = 128,
        use_augmentation: bool = False
    ):
        self.data_root = Path(data_root)
        self.sampling_rate = sampling_rate
        self.sample_rate = sampling_rate
        self.max_files_per_dir = max_files_per_dir
        self.normalize_wav = normalize_wav
        self._base_seed = seed
        self.output_format = output_format
        self.target_frames = target_frames
        self.use_augmentation = use_augmentation

        rng = random.Random(seed)

        items: List[str] = []
        all_dirs = [self.data_root] + [p for p in self.data_root.rglob("*") if p.is_dir()]

        for folder in all_dirs:
            wav_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
            if not wav_files:
                continue

            if len(wav_files) > self.max_files_per_dir:
                wav_files = rng.sample(wav_files, self.max_files_per_dir)

            items.extend([str(p) for p in wav_files])

        if not items:
            raise RuntimeError(f"No wav files found under: {data_root}")

        self.items = sorted(items)

        # 1. MFCC Extractor - Identical settings to the main dataset
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sampling_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 1024, "hop_length": 320, "n_mels": 64, "center": False}
        )

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

    @staticmethod
    def downsample_time(x: torch.Tensor, target_frames: int):
        # x shape: [Channels, Coeffs, Time]
        x = x.unsqueeze(0)  # [1, C, M, T]
        x = F.interpolate(
            x,
            size=(x.shape[2], target_frames),
            mode="bilinear",
            align_corners=False,
        )
        return x.squeeze(0)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        wav_path = self.items[idx]

        try:
            wav, sr = torchaudio.load(wav_path)
            if wav.numel() == 0:
                raise ValueError("Empty audio file")

            if wav.ndim > 1:
                wav = wav.mean(dim=0)
            else:
                wav = wav.squeeze(0)

            wav = torchaudio.functional.highpass_biquad(wav, sr, cutoff_freq=1500)

            if sr != self.sampling_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)
            if self.normalize_wav:
                max_val = wav.abs().max()
                if max_val > 1e-8:
                    wav = wav / max_val
                else:
                    raise ValueError("Signal lost after high-pass filtering")

            if wav.numel() < 1024:
                wav = F.pad(wav, (0, 1024 - wav.numel()))

            speed_factor = random.uniform(0.7, 1.5)
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                wav.unsqueeze(0), sr, [['speed', str(speed_factor)], ['rate', str(self.sample_rate)]]
            )
            wav = wav.squeeze(0)
            mfcc = self.mfcc_transform(wav) 
            delta = torchaudio.functional.compute_deltas(mfcc)
            delta2 = torchaudio.functional.compute_deltas(delta)

            x = torch.stack([mfcc, delta, delta2], dim=0)

            if self.use_augmentation:
                x = self.freq_mask(x)
                x = self.time_mask(x)

            x = self.downsample_time(x, self.target_frames)

            if self.output_format != "cnn":
                x = x.permute(0, 2, 1) 

            # Label 0 for ControlSet
            y = torch.tensor(0, dtype=torch.long)
            return x, y

        except Exception as e:
            print(f"skipping: {wav_path} | Reason: {e}")
            new_idx = random.randint(0, len(self.items) - 1)
            return self.__getitem__(new_idx)