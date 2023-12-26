import torch


ori = torch.Tensor([[1,0,1,1,0],[0,1,1,0,0],[0,0,0,1,1],[1,1,1,1,1]])
ori = ori.unsqueeze(0).to(torch.int64)
print(ori.dtype)
ms = torch.full(ori.shape, 0.5)

cc = torch.stack([ori, ms], dim=1)

print(ori.shape, ms.shape, cc.shape)
print(cc)
print(cc.dtype)

ts = torch.Tensor([[1.,0.5],[0.,0.5],[0.,0.5],[1.,0.5],[1.,0.5]])
sort = torch.sort(ts, 1, descending=True).indices.to(torch.float32)
print(sort.dtype)
print(sort)


