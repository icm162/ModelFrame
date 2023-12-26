tp = (1,2,3,4,5)

def f(*args):
    rt = tuple([i + 5 for i in args])
    return 100, rt

print(*f(*tp))

print(*[1,2,2,2,3,45,5,6,6,6,56])