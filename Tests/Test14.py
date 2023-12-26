import torch

a = torch.Tensor([[1,1,1], [2,3,4]])
b = torch.Tensor([[0.6, 0.3, 0.2]])

print(a*b)
print(torch.mul(a,b))
