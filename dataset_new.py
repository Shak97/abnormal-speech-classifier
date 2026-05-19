import os, random, sys
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

torchaudio.set_audio_backend("soundfile")


def _rglob_filtered(root: str, exts: Tuple[str, ...], ignore_pattern: str = "IfasongIthinkisaLullaby") -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts) and ignore_pattern not in fn:
                out.append(os.path.join(dp, fn))
    out.sort()
    return out


class AudioOnlyRandomChunkDataset(Dataset):
    def __init__(
        self,
        audio_root: str,
        label_map: Optional[Dict[str, int]] = None,
        chunk_seconds: float = 6.0,
        sample_rate: int = 16000,
        normalize_wav: bool = True,
        seed: int = 0,
        n_mfcc: int = 40,
        output_format: str = "cnn",
        target_frames: int = 128,
        use_augmentation: bool = True 
    ):
        self.audio_root = audio_root
        self.chunk_seconds = chunk_seconds
        self.sample_rate = sample_rate
        self.normalize_wav = normalize_wav
        self._base_seed = seed
        self.output_format = output_format
        self.target_frames = target_frames
        self.use_augmentation = use_augmentation

        self.audio_files = _rglob_filtered(audio_root, (".wav", ".flac", ".mp3"))
        
        if not self.audio_files:
            raise RuntimeError(f"No audio files found in {audio_root}")

        self.items = []
        raw_labels = []
        for path in self.audio_files:
            label_str = os.path.basename(os.path.dirname(path)) 
            self.items.append((path, label_str))
            raw_labels.append(label_str)

        if label_map is None:
            uniq = sorted(set(raw_labels))
            self.label_map = {lab: i for i, lab in enumerate(uniq)}
        else:
            self.label_map = label_map

        self.chunk_len = int(self.chunk_seconds * self.sample_rate)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 1024, "hop_length": 320, "n_mels": 64, "center": False}
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def _rng(self) -> random.Random:
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else info.id
        return random.Random(self._base_seed + wid)
    
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
        wav_path, raw_label = self.items[idx]
        wav, sr = torchaudio.load(wav_path)

        if wav.ndim > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)
        if wav.numel() == 0:
            new_idx = (idx + 1) % len(self.items)
            return self.__getitem__(new_idx)

        # hiughpass filter suppreses microphoene noise
        wav = torchaudio.functional.highpass_biquad(wav, sr, cutoff_freq=80.0)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # Random chunking
        if wav.numel() > self.chunk_len:
            start = self._rng().randint(0, wav.numel() - self.chunk_len)
            wav = wav[start:start + self.chunk_len]
        else:
            wav = F.pad(wav, (0, self.chunk_len - wav.numel()))

        if self.normalize_wav:
            wav = wav / wav.abs().max().clamp(min=1e-8)

        if self.use_augmentation and random.random() < 0.7:
            noise = torch.randn_like(wav)
            snr_db = random.uniform(10.0, 30.0)
            noise_gain = wav.abs().max() / (10 ** (snr_db / 20.0))
            wav = wav + noise * noise_gain
            # Re-normalize to prevent clipping
            wav = wav / wav.abs().max().clamp(min=1e-8)

        if self.use_augmentation:
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
            x = x.permute(0, 2, 1) # [C, T, M]
        
        # Label override for abnormal
        label_id = torch.tensor(1, dtype=torch.long)

        return x, label_id
    
def collate_fixed_wave(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0) 
    y = torch.stack(ys, dim=0)
    return x, y