import torch

a = torch.ones((5, 5))
b = torch.ones((5, 5))


print((a - b).sum())

ls = [1,2,3,4,5,6,7,8,9,4]

print(4 in ls, 10 in ls)

ls.remove(4)

print(ls)

print(ls[:0] + ls[1:])

print(ls[:0])

def func(ls):
    ls.remove(4)

a = [1,2,3,4,4,4,5,6,7,8]

func(a[:])

print(a)

bs = [1,2,3,4,5]
print(f"bs: {sum(bs)}")
