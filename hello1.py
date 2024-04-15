import numpy as np
import torch
from icecream import ic

x_free: torch.Tensor = torch.rand(20, 3, requires_grad=True)
x_fixed: torch.Tensor = torch.rand(10, 3, requires_grad=False)
optimizer = torch.optim.Adam([x_free], lr=0.1)
free_idx = np.random.choice(30, size=(20,), replace=False)
free_mask = np.zeros((30,), dtype=bool)
free_mask[free_idx] = True
fixed_mask = ~free_mask
print(free_idx)
for i in range(1000):
    optimizer.zero_grad()
    x: torch.Tensor = torch.zeros(30, 3)
    x[free_mask] = x_free
    x[fixed_mask] = x_fixed
    loss = torch.norm(x)
    loss.backward()
    ic(loss)
    optimizer.step()
print(x_free)
