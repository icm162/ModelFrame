call_space = 100

def makespace(space:int=0, sch:str="-", cs:int=call_space, title:str=None):
    """装饰器 在函数执行前后加入分隔 便于观察输出 1 - 分隔线上下空行数 2 - 分隔线采用字符 3 - 分隔线字符数"""
    def tofunc(f):
        def topara(*args, **kargs):
            print("\n"*space, end="")
            print(cs * sch)
            print("\n"*space, end="")
            if(title != None):
                print(title)
                print("\n"*space, end="")
                print(cs * sch)
                print("\n"*space, end="")
            rst = f(*args, **kargs)
            print("\n"*space, end="")
            print(cs * sch)
            print("\n"*space, end="")
            return rst
        return topara
    return tofunc

@makespace(space=1)
def tprint(text:str, title:str="数据", suffix:str="  "):
    """装饰函数 对打印单行数据增加标题并格式化"""
    title += "：" if(title != "") else ""
    print(title, end=suffix)
    print(text)

def oprint(multext:str, title:str="数据", suffix:str="\n\n"):
    """装饰函数 对打印多行数据增加标题并格式化"""
    tprint(multext, title=title, suffix="\n\n")

def cprint(s:str, color="default"):
    if(color in ("error", "red", "fatal")): print(f"\033[31m{s}\033[0m")
    elif(color in ("healthy", "green", "pass")): print(f"\033[32m{s}\033[0m")
    elif(color in ("info", "blue", "hint")): print(f"\033[34m{s}\033[0m")
    elif(color in ("warn", "yellow", "unhealthy")): print(f"\033[33m{s}\033[0m")
    elif(color in ("unwill", "purplered")): print(f"\033[35m{s}\033[0m")
    else: print(s)

if(__name__ == "__main__"):
    tprint("rua", "")
    cprint("rua", "info")
    cprint("rua", "warn")