import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from third_party.TranAD.src.models import TranAD

# ── 1. Synthetic multivariate time series ────────────────────────────────────
n_train   = 500   # number of timesteps for training
n_test    = 100   # number of timesteps for testing
n_feats   = 4     # number of variables (must be even for TranAD's nhead)

np.random.seed(0)
train_data = np.random.randn(n_train, n_feats)
test_data  = np.random.randn(n_test,  n_feats)

# inject a few anomalies into the test set
test_data[30:35] += 5.0

# ── 2. Convert to sliding windows ────────────────────────────────────────────
def convert_to_windows(data, w_size):
    data = torch.DoubleTensor(data)
    windows = []
    for i in range(len(data)):
        if i >= w_size:
            w = data[i - w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w)
    return torch.stack(windows)

# ── 3. Build model ────────────────────────────────────────────────────────────
model     = TranAD(n_feats).double()
w_size    = model.n_window   # default window size from TranAD

trainD = convert_to_windows(train_data, w_size)
testD  = convert_to_windows(test_data,  w_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
loss_fn   = nn.MSELoss(reduction='none')

# ── 4. Training loop ──────────────────────────────────────────────────────────
n_epochs = 5
print("Training TranAD...")
for epoch in range(n_epochs):
    model.train()
    dataset    = TensorDataset(trainD, trainD)
    dataloader = DataLoader(dataset, batch_size=model.batch)
    n = epoch + 1
    losses = []
    for d, _ in dataloader:
        local_bs = d.shape[0]
        window   = d.permute(1, 0, 2)
        elem     = window[-1, :, :].view(1, local_bs, n_feats)
        z        = model(window, elem)
        # adversarial loss weighting shifts each epoch
        l1 = (1 / n) * loss_fn(z[0], elem) + (1 - 1 / n) * loss_fn(z[1], elem)
        loss = torch.mean(l1)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()
    print(f"  Epoch {epoch}  loss={np.mean(losses):.6f}")

# ── 5. Inference ──────────────────────────────────────────────────────────────
print("\nTesting TranAD...")
model.eval()
with torch.no_grad():
    dataset    = TensorDataset(testD, testD)
    dataloader = DataLoader(dataset, batch_size=len(testD))
    for d, _ in dataloader:
        window = d.permute(1, 0, 2)
        elem   = window[-1, :, :].view(1, len(testD), n_feats)
        z      = model(window, elem)
        scores = loss_fn(z[1], elem)[0].mean(dim=-1).numpy()  # anomaly score per timestep

# ── 6. Results ────────────────────────────────────────────────────────────────
print("\nAnomaly scores (first 40 timesteps):")
for i, s in enumerate(scores[:40]):
    flag = " ← anomaly injected" if 30 <= i < 35 else ""
    print(f"  t={i:3d}  score={s:.4f}{flag}")
