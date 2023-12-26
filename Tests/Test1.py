ls = [1,2,3,4,5,6,7,8]


print(ls)
print(list(reversed(ls)))
print(ls)

# for i in reversed(ls): print(i)


for i in range(0,-1,-1):
    print(i)

# a = [1,2]
# b = [3,4]
# cc = [[-1, 0]]
# print(a, b, cc)
# cc.append(a.extend(b))
# print(cc)
# print(a, b, cc)
# print(a + b) 

ls = [[1,6,10,13,15], [2,7,11,14], [3, 8, 12], [4, 9], [5]]
lss = [[1,2,3,4,5], [6,7,8,9], [10,11,12], [13,14], [15]]

import numpy as np

alss = np.array(lss)


print(lss[1][:2])

