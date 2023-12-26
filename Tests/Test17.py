import re


str = "test_1.png"

print(re.findall("\d+", str)[0])
