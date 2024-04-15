import subprocess
import time
import pyautogui
import pygetwindow as gw
import argparse

def runcap(exe_path, window_title, action_delay=5, screenshot_path="screenshot.png"):
    # 启动exe文件
    print("launching exe ...")
    subprocess.Popen(exe_path)

    print("waiting for window ...")
    # 等待窗口加载
    time.sleep(3)  # 根据程序启动速度调整等待时间

    # 确保窗口已经出现
    win = None
    while win is None:
        windows = gw.getWindowsWithTitle(window_title)
        if windows:
            win = windows[0]
            win.activate()  # 激活窗口以确保它在前台
        time.sleep(1)

    # 等待额外的延时以确保应用程序已经完全加载
    time.sleep(action_delay)

    # 在应用程序中执行动作，例如点击按钮
    pyautogui.click(x=100, y=200)  # 根据需要调整坐标

    # 截图
    pyautogui.screenshot(screenshot_path)

    # 可以添加更多动作或处理逻辑
    # ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an EXE and capture a screenshot of its window.")
    parser.add_argument("exe_path", type=str, help="The path to the executable")
    parser.add_argument("window_title", type=str, help="The title of the window to capture")
    parser.add_argument("--action_delay", type=int, default=5, help="Time to wait before taking action")
    parser.add_argument("--screenshot_path", type=str, default="screenshot.png", help="Path to save the screenshot")

    args = parser.parse_args()

    runcap(args.exe_path, args.window_title, args.action_delay, args.screenshot_path)
