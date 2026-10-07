from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.star_ml import StarPatchCNN, MODEL_FILE


def make_data(n=6000,size=32,seed=2026):
    rng=np.random.default_rng(seed)
    x=np.zeros((n,1,size,size),np.float32); y=np.zeros(n,np.float32)
    yy,xx=np.mgrid[:size,:size]
    for i in range(n):
        if i<n//2:
            y[i]=1
            cx=rng.uniform(size*0.35,size*0.65); cy=rng.uniform(size*0.35,size*0.65)
            sigma=rng.uniform(0.7,2.0); strength=rng.uniform(0.5,1.0)
            base=rng.uniform(0.0,0.22,(size,size))
            star=strength*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma*sigma))
            x[i,0]=np.clip(base+star,0,1)
        else:
            base=rng.uniform(0.0,0.28,(size,size))
            # negative patches can contain gradients, hot pixels, and elongated streaks
            if rng.random()<0.35:
                bx=rng.integers(3,size-3); by=rng.integers(3,size-3)
                base[max(0,by-1):by+2,max(0,bx-1):bx+2]+=rng.uniform(0.1,0.8)
            x[i,0]=np.clip(base,0,1)
    idx=rng.permutation(n); return x[idx],y[idx]

def main():
    torch.manual_seed(2026); np.random.seed(2026)
    x,y=make_data()
    split=int(0.8*len(y))
    train=DataLoader(TensorDataset(torch.from_numpy(x[:split]),torch.from_numpy(y[:split])),batch_size=128,shuffle=True)
    val=DataLoader(TensorDataset(torch.from_numpy(x[split:]),torch.from_numpy(y[split:])),batch_size=128)
    m=StarPatchCNN(); opt=torch.optim.Adam(m.parameters(),lr=2e-3); lossfn=nn.BCEWithLogitsLoss()
    best=0
    for epoch in range(1,5):
        m.train();
        for xb,yb in train:
            opt.zero_grad(); loss=lossfn(m(xb),yb); loss.backward(); opt.step()
        m.eval(); correct=total=0
        with torch.no_grad():
            for xb,yb in val:
                pred=(torch.sigmoid(m(xb))>0.5).float(); correct+=(pred==yb).sum().item(); total+=len(yb)
        acc=correct/total; print(f"epoch {epoch}: val_acc={acc:.3f}")
        if acc>=best:
            best=acc; Path(MODEL_FILE).parent.mkdir(exist_ok=True); torch.save(m.state_dict(),MODEL_FILE)
    print(f"saved {MODEL_FILE} with val_acc={best:.3f}")
if __name__=='__main__': main()
