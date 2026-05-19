import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models


from dataset_new import collate_fixed_wave, AudioOnlyRandomChunkDataset
from dataset_controlset_new import ControlSetDataset
import torchaudio
torchaudio.set_audio_backend("soundfile")

def build_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    model = models.mobilenet_v2(pretrained=pretrained)
    # change input channels from 3 -> 1
    # model.features[0][0] = nn.Conv2d(
    #     1, 32, kernel_size=3, stride=2, padding=1, bias=False
    # )
    # change classifier output
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    # logits: [B, C], y: [B]
    pred = logits.argmax(dim=1)
    correct = (pred == y).sum().item()
    return correct / max(1, y.numel())


def move_to_device(x, y, device):
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler = None,
    log_every: int = 50,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    t0 = time.time()
    for step, (x, y) in enumerate(loader):
        x, y = move_to_device(x, y, device)

        optimizer.zero_grad(set_to_none=True)

        # AMP optional
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        acc = accuracy_from_logits(logits, y)

        running_loss += loss.item()
        running_acc += acc
        n_batches += 1

        if log_every and (step + 1) % log_every == 0:
            dt = time.time() - t0
            print(
                f"  step {step+1:>5}/{len(loader)} | "
                f"loss {running_loss/n_batches:.4f} | "
                f"acc {running_acc/n_batches:.4f} | "
                f"time {dt:.1f}s"
            )

    return {
        "loss": running_loss / max(1, n_batches),
        "acc": running_acc / max(1, n_batches),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = move_to_device(x, y, device)
        logits = model(x)
        loss = criterion(logits, y)
        acc = accuracy_from_logits(logits, y)

        running_loss += loss.item()
        running_acc += acc
        n_batches += 1

    return {
        "loss": running_loss / max(1, n_batches),
        "acc": running_acc / max(1, n_batches),
    }


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    num_workers: int = 0
    val_ratio: float = 0.2
    seed: int = 1234
    use_amp: bool = True
    save_dir: str = "./checkpoint_synthetic_control"
    save_name: str = "mobilenetv2_logmel_best.pt"


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(ds, ds_control, collate_fn, num_classes: int, cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    n_total = len(ds) + len(ds_control)
    n_val = int(n_total * cfg.val_ratio)
    n_train = n_total - n_val
    train_ds = torch.utils.data.ConcatDataset([ds, ds_control])

    train_ds, val_ds = random_split(
        train_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    print("cfg.num_workers =", cfg.num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    print("train_loader.num_workers =", train_loader.num_workers)
    print("val_loader.num_workers =", val_loader.num_workers)

    model = build_model(num_classes=num_classes, pretrained=True).to(device)

    weights = torch.tensor([167.36, 0.50], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if (cfg.use_amp and device.type == "cuda") else None

    best_val_acc = -1.0
    best_path = os.path.join(cfg.save_dir, cfg.save_name)

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")

        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            log_every=50,
        )
        va = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"  train: loss {tr['loss']:.4f} | acc {tr['acc']:.4f}\n"
            f"  val:   loss {va['loss']:.4f} | acc {va['acc']:.4f}"
        )

        # Save best
        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "label_map": getattr(ds, "label_map", None),
                },
                best_path,
            )
            print(f"saved best to: {best_path} (val_acc={best_val_acc:.4f})")

    print("\nDone. Best val acc:", best_val_acc)



if __name__ == "__main__":
    AUDIO_ROOT = r"/media/oem/storage01/Shakeel/korean_speech/dataset/speech_abnormal_dataset/train"
    # JSON_ROOT  = r"D:\server\speechprocessing\aihubdata\training\label\TL01_뇌신경장애"
    audio_root_control_set = r'/media/oem/storage01/Shakeel/korean_speech/dataset/audio_control_dataset/train'
    audio_root_control_set = r'/media/oem/storage01/Shakeel/korean_speech/dataset/fishthisbish/synthetic_control_set/'

    ds_control = ControlSetDataset(
        data_root=audio_root_control_set,
        max_files_per_dir=50,
        n_mfcc=40,
        sampling_rate=16000,
        target_frames=128,
    )

    ds = AudioOnlyRandomChunkDataset(
        audio_root=AUDIO_ROOT,
        label_map=None,
        chunk_seconds=6.0,
        sample_rate=16000,
        seed=1234,
        output_format="cnn",
        target_frames=128,
        n_mfcc=40, 
        use_augmentation=True
    )

    num_classes = len(ds.label_map)
    print("num_classes:", num_classes)
    print("label_map:", ds.label_map)

    cfg = TrainConfig(
        epochs=20,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        num_workers=4,
        val_ratio=0.2,
        use_amp=True,
    )

    main(ds, ds_control, collate_fixed_wave, num_classes=2, cfg=cfg)
