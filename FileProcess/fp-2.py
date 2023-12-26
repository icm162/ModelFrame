import shutil
import os

path = r"F:\DAT\PH2\val\label"
to_path = r"F:\PTC\UNetSEG\Cond\PH2\Label"

for i, file_name in enumerate(os.listdir(path)):
    shutil.copy(os.path.join(path, file_name), os.path.join(to_path, f"{i + 1}.png"))
