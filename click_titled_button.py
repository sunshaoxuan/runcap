import os
import subprocess
import time
import pyautogui
import pygetwindow as gw
import argparse
import def_funcions
from pywinauto.application import Application
import inspect

def runcap(window_title, button_title):
    # 获取主窗口
    app = Application(backend="uia").connect(title=window_title)
    dlg = app.window(title=window_title)

    # 查找名为"xxx"的按钮并点击
    # 你可以使用按钮的名字、自动化ID、控件类型等不同属性来定位它
    print("find and click login button")
    button = dlg.child_window(title=button_title, control_type="Button")
    button.click()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clich a titled button in a window")
    parser.add_argument("window_title", type=str, help="The title of the window when the executable program was running")
    parser.add_argument("button_title", type=str, help="The title of the button")
    args = parser.parse_args()
    runcap(args.window_title, args.button_title)
