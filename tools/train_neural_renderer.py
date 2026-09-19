"""Reproduce the tiny M7 neural-state-conditioned appearance network.

This is intentionally a feasibility model, not a NeRF and not a biological vision
model. It learns a synthetic RGB appearance mapping conditioned on four neural
summary values (motion, looming/escape, ON, OFF). The trained weights are baked
into src/main_feasibility.cpp so inference has no runtime ML dependency.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 12000
rgb = rng.random((N, 3))
neural = rng.random((N, 4))
X = np.concatenate([rgb, neural], axis=1)
motion, looming, on, off = neural.T

lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
gray = lum[:, None]
target = gray + (rgb - gray) * (1 + 0.35 * motion)[:, None]
target[:, 0] += 0.18 * looming + 0.05 * on
target[:, 1] += 0.08 * motion
target[:, 2] += 0.12 * off - 0.08 * looming
target = 1 / (1 + np.exp(-4 * (target - 0.5)))
target = np.clip(target, 0, 1)

H = 10
W1 = rng.normal(0, 0.3, (7, H))
b1 = np.zeros(H)
W2 = rng.normal(0, 0.3, (H, 3))
b2 = np.zeros(3)

for _ in range(2500):
    idx = rng.integers(0, N, 512)
    x, y = X[idx], target[idx]
    h = np.tanh(x @ W1 + b1)
    yhat = 1 / (1 + np.exp(-(h @ W2 + b2)))
    d = (yhat - y) * (2 / len(idx))
    dz2 = d * yhat * (1 - yhat)
    gW2 = h.T @ dz2
    gb2 = dz2.sum(0)
    dz1 = (dz2 @ W2.T) * (1 - h * h)
    W1 -= 0.04 * (x.T @ dz1)
    b1 -= 0.04 * dz1.sum(0)
    W2 -= 0.04 * gW2
    b2 -= 0.04 * gb2

pred = 1 / (1 + np.exp(-(np.tanh(X @ W1 + b1) @ W2 + b2)))
print("MSE:", np.mean((pred - target) ** 2))
for name, value in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
    print(name, "=", value.tolist())
