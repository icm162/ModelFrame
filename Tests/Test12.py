def split(index, split=[8, 2]):
    sub = "val"
    assert len(split) == 2, "数据集划分比例有误"
    summary, no = sum(split), 0 if(sub == "train") else 1 if(sub == "val") else -1
    return (index // split[no]) * summary + (index % split[no]) + (0 if(sub == "train") else split[0])

for i in range(20):
    print(split(i), end = "  ")