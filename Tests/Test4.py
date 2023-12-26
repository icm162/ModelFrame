from typing import Optional, Union

def func(a:Union[float, int, str]):
    print(type(a))
    print(a)


func(a=1.90)
func(a=2)
func(a="rua")
func(a=[2,6])
func()

def func2(a:int):
    print(a)

func2(a="523")
