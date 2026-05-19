import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
torchaudio.set_audio_backend("soundfile")

from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)
import numpy as np
import matplotlib.pyplot as plt
from train_new import build_model
import seaborn as sns

class HubertAbnormalityClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_base=True):
        super().__init__()
        
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.hubert = bundle.get_model()
        
        if freeze_base:
            for param in self.hubert.parameters():
                param.requires_grad = False
                
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        x shape: [Batch_size, Time_samples] (Raw 16kHz audio)
        """
        output, _ = self.hubert(x) 
        pooled_output = output.mean(dim=1) 
        logits = self.classifier(pooled_output)
        
        return logits

@torch.no_grad()
def run_evaluation(model, loader, device, label_names):
    model.eval()
    all_preds = []
    all_labels = []

    print(f"Starting evaluation on {len(loader.dataset)} samples...")
    
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    conf_mat = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=label_names,
        digits=4
    )

    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Overall Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return all_labels, all_preds, conf_mat

def plot_confusion_matrix(conf_mat, label_names, save_path="confusion_matrix.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_mat, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=label_names, 
        yticklabels=label_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # 1. Setup Device and Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/mobilenetv2_logmel_best.pt"
    checkpoint_path = r'checkpoints_fixes/mobilenetv2_logmel_best.pt'
    checkpoint_path = r'checkpoint_synthetic_control/mobilenetv2_logmel_best.pt'
    checkpoint_path = r'checkpoint_synthetic_control/mobilenetv2_logmel_best.pt'
    # checkpoint_path = r'checkpoint_synthetic_control/hubert_classifider.pt'
    
    # 2. Load Checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    label_map = checkpoint.get("label_map", {0: "Normal", 1: "Abnormal"})
    label_map = {0: "Normal", 1: "Abnormal"}
    # Ensure labels are sorted by index for the report
    label_names = [label_map[i] for i in sorted(label_map.keys())]
    num_classes = len(label_names)

    # 3. Initialize and Load Model
    model = build_model(num_classes=2, pretrained=True)
    # model = HubertAbnormalityClassifier(num_classes=2, freeze_base=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    from dataset_new import AudioOnlyRandomChunkDataset, collate_fixed_wave
    from dataset_controlset_new import ControlSetDataset
    
    AUDIO_ROOT = r"/media/oem/storage01/Shakeel/korean_speech/dataset/speech_abnormal_dataset/test"
    # JSON_ROOT  = r"D:\server\speechprocessing\aihubdata\training\label\TL01_뇌신경장애"


    audio_root_control_set = r'/media/oem/storage01/Shakeel/korean_speech/dataset/audio_control_dataset/test'
    audio_root_control_set = r'/media/oem/storage01/Shakeel/korean_speech/dataset/audio_inhouse/normal_split'

    # ds_control = ControlSetDataset(
    #     data_root=audio_root_control_set,
    #     max_files_per_dir=50,
    # )

    # ds = AudioOnlyRandomChunkDataset(
    #     audio_root=AUDIO_ROOT,
    #     label_map=None,
    #     chunk_seconds=10.0,
    #     sample_rate=16000,
    #     # strict_duplicates=False,
    #     seed=1234,
    #     output_format="cnn",     # IMPORTANT for MobileNet
    #     target_frames=128,
    #     n_mels=128,
    # )

    ds_control = ControlSetDataset(
        data_root=audio_root_control_set,
        max_files_per_dir=50,
        n_mfcc=40,
        sampling_rate=16000,
        target_frames=128
    )

    ds = AudioOnlyRandomChunkDataset(
        audio_root=AUDIO_ROOT,
        label_map=None,
        chunk_seconds=6.0,
        sample_rate=16000,
        seed=1234,
        output_format="cnn",
        target_frames=128,
        n_mfcc=40
    )

    test_ds = torch.utils.data.ConcatDataset([ds, ds_control])

    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fixed_wave
    )

    # 5. Execute Evaluation
    y_true, y_pred, cm = run_evaluation(model, test_loader, device, label_names)
    
    # 6. Visualize
    plot_confusion_matrix(cm, label_names)