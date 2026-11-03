"""A small network learns to tell two spirals apart, trained on the device:
tensors, autograd, nn and an optimizer. Nothing is downloaded."""

import math
import time

import torch
from torch import nn

torch.manual_seed(0)


def spirals(count=600, noise=0.05):
    """Two interleaved spirals, one per class."""
    turn = torch.linspace(0.3, 3 * math.pi, count // 2)
    arm = torch.stack([turn * torch.cos(turn), turn * torch.sin(turn)], dim=1) / (3 * math.pi)
    points = torch.cat([arm, -arm]) + noise * torch.randn(count, 2)
    labels = torch.cat([torch.zeros(count // 2), torch.ones(count // 2)]).long()
    order = torch.randperm(count)
    return points[order], labels[order]


points, labels = spirals()
train_x, train_y = points[:480], labels[:480]
test_x, test_y = points[480:], labels[480:]

model = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

print(f"PyTorch {torch.__version__}, {sum(p.numel() for p in model.parameters())} weights")
start = time.perf_counter()
for epoch in range(1, 401):
    optimizer.zero_grad()
    loss = loss_fn(model(train_x), train_y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        with torch.no_grad():
            accuracy = (model(test_x).argmax(dim=1) == test_y).float().mean().item()
        print(f"epoch {epoch:3}  loss {loss.item():.3f}  test accuracy {accuracy:.0%}")
print(f"Trained in {time.perf_counter() - start:.2f} s")
