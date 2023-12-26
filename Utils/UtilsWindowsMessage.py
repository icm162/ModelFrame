import platform
if(platform.system() == "Windows"): 
    import win32gui, win32con

    def destroy():
        pass

    def send_windows_message(title:str, contents:str) -> None:
        windowClass = win32gui.WNDCLASS()
        windowClass.lpszClassName = " "
        windowClass.lpfnWndProc = {win32con.WM_DESTROY: destroy}
        windowClass.hInstance = win32gui.GetModuleHandle(None)
        winHandle = win32gui.CreateWindow(
            win32gui.RegisterClass(windowClass),
            "", 
            win32con.WS_OVERLAPPED, # | win32con.WS_SYSMENU,
            0, 0,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            0, 0,
            windowClass.hInstance,
            None
        )
        nid = (
            winHandle,
            0, # 托盘图标 ID
            win32gui.NIF_INFO, # 标识
            0, # 回调信息 ID
            0, # 托盘图标句柄
            "", # 图标字符串
            contents, # 消息字符串
            0, # 提示显示时间
            title, # 标题字符串
            win32gui.NIIF_INFO # 提示用到的图标
        )
        win32gui.Shell_NotifyIcon(
            win32gui.NIM_ADD,
            nid
        )
        win32gui.PostQuitMessage(0)
        destroy()

else:
    def send_windows_message(*args):
        pass

if(__name__ == "__main__"):
    send_windows_message("你好", "你好")




