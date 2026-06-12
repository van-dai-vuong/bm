"""
Example: anomaly detection on a multivariate time series using the
tranad library — using backprop and convert_to_windows from tranad.src.main.
Run:  python examples/example_tranad.py
"""
import numpy as np
import torch
from tranad.src.models import TranAD
from tranad.src.main import backprop, convert_to_windows   # ← reuse library internals

# ── 1. Synthetic multivariate time series ────────────────────────────────────
n_train, n_test, n_feats = 500, 100, 4
np.random.seed(0)
train_np = np.random.randn(n_train, n_feats)
test_np  = np.random.randn(n_test,  n_feats)
test_np[30:35] += 5.0                           # inject anomalies

# TranAD expects DoubleTensors (float64)
trainO = torch.DoubleTensor(train_np)           # raw originals — kept for backprop's dataO arg
testO  = torch.DoubleTensor(test_np)

# ── 2. Model ──────────────────────────────────────────────────────────────────
model = TranAD(n_feats).double()

# ── 3. Sliding windows via TranAD's own convert_to_windows(data, model) ───────
# mirrors main.py:  trainD = convert_to_windows(trainD, model)
trainD = convert_to_windows(trainO, model)      # [n_train, w_size, n_feats]
testD  = convert_to_windows(testO,  model)      # [n_test,  w_size, n_feats]

# ── 4. Optimizer + scheduler (same defaults as main.py) ──────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

# ── 5. Training via backprop ──────────────────────────────────────────────────
# mirrors main.py:  lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
n_epochs = 5
accuracy_list = []
print("Training TranAD...")
for epoch in range(1, n_epochs + 1):
    lossT, lr = backprop(epoch, model, trainD, trainO, optimizer, scheduler)
    accuracy_list.append((lossT, lr))
    print(f"  Epoch {epoch}  loss={lossT:.6f}  lr={lr:.6f}")

# ── 6. Threshold calibration on training data ────────────────────────────────
# mirrors main.py:  model.eval() then backprop(0, ..., training=False) on TRAIN split
model.eval()
loss_train, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
# loss_train shape: [n_train, n_feats]
threshold_upper = np.nanmax(loss_train[:, 0]) * 1.1
threshold_lower = np.nanmin(loss_train[:, 0]) * 0.9
print(f"\nThresholds  lower={threshold_lower:.4f}  upper={threshold_upper:.4f}")

# ── 7. Inference on test data ─────────────────────────────────────────────────
# mirrors main.py:  loss, y_pred = backprop(0, model, testD, testO, ..., training=False)
loss_test, _ = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
scores = loss_test[:, 0]                        # feature-0 anomaly score per timestep

# ── 8. Results ────────────────────────────────────────────────────────────────
print("\nAnomaly scores (first 40 timesteps):")
for i, s in enumerate(scores[:40]):
    injected = " <- anomaly injected" if 30 <= i < 35 else ""
    detected = " [DETECTED]"          if s > threshold_upper or s < threshold_lower else ""
    print(f"  t={i:3d}  score={s:.4f}{injected}{detected}")