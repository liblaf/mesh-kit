import torch
from jaxtyping import Bool, Float, Integer

B = Bool[torch.Tensor, ""]
BN = Bool[torch.Tensor, "N"]

F = Float[torch.Tensor, ""]
F3 = Float[torch.Tensor, "3"]
F33 = Float[torch.Tensor, "3 3"]
F34 = Float[torch.Tensor, "3 4"]
F4 = Float[torch.Tensor, "4"]
F43 = Float[torch.Tensor, "4 3"]
F44 = Float[torch.Tensor, "4 4"]
FMN = Float[torch.Tensor, "M N"]
FMN3 = Float[torch.Tensor, "M N 3"]
FN = Float[torch.Tensor, "N"]
FN3 = Float[torch.Tensor, "N 3"]
FNN = Float[torch.Tensor, "N N"]

I = Integer[torch.Tensor, ""]  # noqa: E741
I2 = Integer[torch.Tensor, "2"]
I3 = Integer[torch.Tensor, "3"]
I4 = Integer[torch.Tensor, "4"]
IN = Integer[torch.Tensor, "N"]
IN2 = Integer[torch.Tensor, "N 2"]
IN3 = Integer[torch.Tensor, "N 3"]
IN4 = Integer[torch.Tensor, "N 4"]
