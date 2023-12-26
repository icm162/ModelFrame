import os

path = r"F:\PTC\UNetSEG\Cond\Label"

file_ls = sorted(os.listdir(path), key=lambda k: int(k.split(".")[0].split("_")[0]))
delta = int(file_ls[0].split(".")[0].split("_")[0]) - 1
to_ls = list(map(lambda f: os.path.join(path, str(int(f.split(".")[0].split("_")[0]) - delta) + f".{f.split('.')[1]}"), file_ls))
file_ls = list(map(lambda f: os.path.join(path, f), file_ls))

for i, file_path in enumerate(file_ls): os.rename(file_path, to_ls[i])
