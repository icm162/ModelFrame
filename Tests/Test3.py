import UNetSEG.Tests.TestP as p

dic = {1:6, 6:34}

print(isinstance(dic, dict))
print(isinstance(dic, list))

p.a = 6

print(p.a)

p.add(p.b)

print(p.b)

print(3 // 2, 7 // 2)

import re

print(re.match("EP\d+", "EP356") != None)
print(re.match("EP\d+", "BESTEN") != None)

mean = lambda ls: sum(ls) / len(ls)
print(mean([1,6,9,3,8,9]))
