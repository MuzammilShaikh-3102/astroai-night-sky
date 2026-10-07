from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT=Path(__file__).resolve().parents[1]
MODEL_FILE=ROOT/"models"/"star_patch_cnn.pt"

class StarPatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,8,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,24,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head=nn.Sequential(nn.Flatten(), nn.Linear(24,16), nn.ReLU(), nn.Linear(16,1))
    def forward(self,x): return self.head(self.net(x)).squeeze(1)

def load_model():
    if not MODEL_FILE.exists(): return None
    m=StarPatchCNN()
    try:
        m.load_state_dict(torch.load(MODEL_FILE,map_location="cpu")); m.eval(); return m
    except Exception: return None

def score_patch(model, patch: np.ndarray) -> float | None:
    if model is None: return None
    x=torch.from_numpy(patch.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return float(torch.sigmoid(model(x))[0])
