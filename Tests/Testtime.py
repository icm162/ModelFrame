import time

stamp = time.localtime(1668429811)

print(f"{time.timezone // 3600}  {stamp.tm_year}/{stamp.tm_mon}/{stamp.tm_mday} {stamp.tm_hour}:{stamp.tm_min}:{stamp.tm_sec}")