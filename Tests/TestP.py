a = 1
b = 2

def add(s):
    s += 10
    
a, b = a+b,6
print(a,b)


class car:

    def __init__(self, wheel_num:int):
        self.wheel_num = wheel_num

    def get_wheels(self):
        return self.wheel_num

    
clazz = car

panamera = clazz(4)

print(panamera.get_wheels())

import torch
import numpy as np

a = np.array([[[1, 0], [0, 1]], [[3, 2], [5, 6]]])

t = torch.Tensor(a)

print(t[:,0,:].shape)
print(t)
print(t.argmax(1).unsqueeze(1))
print(t.argmax(1).unsqueeze(1).shape)

