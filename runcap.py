import os
import subprocess
import time
import pyautogui
import pygetwindow as gw
from pygetwindow import PyGetWindowException
import argparse
from pywinauto.application import Application
from pywinauto.mouse import double_click
from pywinauto.timings import Timings
from pywinauto.findwindows import ElementNotFoundError
from PIL import Image, ImageGrab, ImageChops, ImageFilter, ImageEnhance
import pytesseract
from collections import namedtuple
import inspect
import ctypes
from ctypes import wintypes
import numpy as np
import win32api
import win32gui
import win32con

# 加载用户32库
user32 = ctypes.windll.user32

def find_window_by_class(class_name):
    # 尝试找到具有给定类名的窗口
    return ctypes.windll.user32.FindWindowW(class_name, None)

def toggle_desktop_icons(show):
    # 获取Progman窗口句柄
    progman = find_window_by_class("Progman")
    # 尝试在Progman下找到SHELLDLL_DefView窗口
    def_view = ctypes.windll.user32.FindWindowExW(progman, None, "SHELLDLL_DefView", None)
    if not def_view:
        # 在WorkerW中寻找
        workerw = None
        while True:
            workerw = ctypes.windll.user32.FindWindowExW(None, workerw, "WorkerW", None)
            if not workerw:
                break
            def_view = ctypes.windll.user32.FindWindowExW(workerw, None, "SHELLDLL_DefView", None)
            if def_view:
                break

    if def_view:
        ctypes.windll.user32.ShowWindow(def_view, 1 if show else 0)

def hide_desktop_icons():
    toggle_desktop_icons(False)

def show_desktop_icons():
    toggle_desktop_icons(True)

def minimize_all_windows():
    pyautogui.hotkey('win', 'd')

def set_english_input():
    # 获取当前窗口句柄
    hwnd = win32gui.GetForegroundWindow()
    
    # 加载默认英文输入法
    hkl_english = win32api.LoadKeyboardLayout('00000409', win32con.KLF_ACTIVATE)

    # 设置当前窗口使用英文输入法
    win32api.SendMessage(hwnd, win32con.WM_INPUTLANGCHANGEREQUEST, 0, hkl_english)

# 获取当前光标
def get_cursor():
    cur = wintypes.HCURSOR()
    cur = user32.GetCursor()
    return cur

# 检测光标是否为忙碌状态
def is_busy_cursor():
    # 定义忙碌光标的标准句柄（通常是沙漏或旋转圈）
    IDC_WAIT = user32.LoadCursorW(0, 32514)
    current_cursor = get_cursor()
    return current_cursor == IDC_WAIT

def confirm_window(window_title):
    # 确保窗口已经出现
    win = None
    while win is None:
        windows = gw.getWindowsWithTitle(window_title)
        if windows:
            print(f"找到窗口: {window_title}")
            time.sleep(1)
            win = windows[0]

            # print(f"{window_title} members:")
            # members = inspect.getmembers(win)
            # for name, value in members:
            #    print(f"{name}: {value}")
            
            try:
                if not win.isActive:
                    time.sleep(2)
                    win.activate()  # 激活窗口以确保它在前台
            except PyGetWindowException as e:
                print(f"{window_title}: {e}")
    return win

def screen_has_stabilized(prev_image, current_image, threshold=5):
    # 计算两张图片之间的差异
    diff = ImageChops.difference(prev_image, current_image)
    # 将差异转换为数组并计算非零元素的数量
    np_diff = np.array(diff)
    change_count = np.count_nonzero(np_diff)
    # 如果变化的像素小于阈值，则认为屏幕稳定
    return change_count < threshold

def capture_when_stable(shotRegion=None, wait_time=2, stability_threshold=5, max_attempts=30):
    prev_image = None
    stable_count = 0
    
    for _ in range(max_attempts):
        if shotRegion == None:
            current_image = pyautogui.screenshot()
        else:
            current_image = pyautogui.screenshot(region=shotRegion)
        if prev_image is not None:
            if screen_has_stabilized(prev_image, current_image, threshold=stability_threshold):
                stable_count += 1
            else:
                stable_count = 0  # 重置稳定计数器
        if stable_count >= wait_time:  # 连续稳定wait_time次后认为屏幕已稳定
            return current_image
            break
        prev_image = current_image
        time.sleep(0.2)  # 等待0.2秒
        
def screen_shot(win, dir_path, file_name, num, adj, is_full_screen = False):
    Rectangle = namedtuple('Rectangle', 'left top right bottom')
    if not is_full_screen:
        # 获取窗口的位置和大小
        adjnum = 2
        if adj:
            adjnum = 7

        x, y, width, height = win.left, win.top, win.width, win.height
        
        shotRegion = Rectangle(x + adjnum, y + 1, width - 2 * adjnum, height - 1 - adjnum)
        screenshot = capture_when_stable(shotRegion)
    else:
        screenshot = capture_when_stable()

    screenshot_path = os.path.join(dir_path, "output", f"{file_name}{num}.png")
    screenshot.save(screenshot_path)
    print("保存截图: " + screenshot_path)
    return screenshot_path

def find_next_edit_by_rect(dlg_win, label_rect):
    # 获取所有文本框
    edit_boxes = dlg_win.descendants(control_type='Edit')

    # 初始设置 edit_box 为 None
    edit_box = None

    # 找到位于标签右侧的第一个文本框，且其中心线穿过标签的垂直范围
    for edit in edit_boxes:
        edit_rect = edit.rectangle()
        edit_center_y = (edit_rect.top + edit_rect.bottom) / 2

        # 检查是否右侧并且中心线在标签的垂直范围内
        if edit_rect.left > label_rect.right and label_rect.top <= edit_center_y <= label_rect.bottom:
            edit_box = edit
            break

    return edit_box


def find_rect_by_text_ocr(image_path, keyword, language, left_offset = 0, top_offset = 0):
    # 设置 Tesseract-OCR 的路径
    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

    # 加载图像文件
    # 加载图像并转换为灰度
    img = Image.open(image_path).convert('L')

    # 应用阈值处理
    threshold = 200
    img = img.point(lambda p: p > threshold and 255)

    # 应用锐化滤镜
    img = img.filter(ImageFilter.SHARPEN)

    # 增强对比度
    enhancer = ImageEnhance.Contrast(img)       # 增强对比度
    img = enhancer.enhance(2)

    # 使用 pytesseract 进行 OCR，指定语言和PSM模式
    custom_config = r'--oem 3 --psm 11 -l ' + language
    data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)

    # print(pytesseract.image_to_string(img, language))

    # 解析返回的数据，找到关键字的位置
    found = False
    rect = None
    Rectangle = namedtuple('Rectangle', 'left top right bottom')
    for i in range(len(data['text'])):
        if data['text'][i].strip() == keyword:  # 使用 strip() 来去除可能的前后空白字符
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            rect = Rectangle(x + left_offset, y + top_offset, x + left_offset + w, y + top_offset + h)
            print(f"Found '{keyword}' at position: ({x}, {y}), size: ({w}x{h})")
            found = True
            break

    if not found:
        print(f"'{keyword}' not found in the image.")

    return rect

def find_edit_by_label(dlg_win, label_value, image_path):
    # 遍历所有Edit控件并检查Value属性
    found_label = None
    for label in dlg_win.descendants(control_type="Edit"):
        # print(label.iface_value.value)
        try:
            # 尝试获取值，如果支持 ValuePattern
            current_value = label.legacy_properties()["Value"]
            if current_value == label_value:
                found_label = label
                break
        except Exception as e:
            print("此编辑框不支持 ValuePattern 或其他错误发生:", str(e))
    
    label_rect = None

    if found_label == None:
        # 获取窗体在屏幕上的位置
        dlg_rect = dlg_win.rectangle()
        label_rect = find_rect_by_text_ocr(image_path, label_value, "jpn", dlg_rect.left, dlg_rect.top)
        if label_rect == None:
            return None
    else:
        # 获取标签的位置信息
        label_rect = found_label.rectangle()

    # print(f"label_value: {label_value}")
    # print(f"current_value: {current_value}")
    # print(f"label: {label_rect.left},{label_rect.top},{label_rect.right},{label_rect.bottom}")
    return find_next_edit_by_rect(dlg_win, label_rect)

def runcap(exe_path, window_title, action_delay=3, screenshot_name="screenshot"):
    set_english_input()
    minimize_all_windows()
    hide_desktop_icons()

    i = 0
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    # 获取当前脚本的绝对路径
    program_path = os.path.abspath(exe_path)

    # 获取脚本所在的目录
    program_dir = os.path.dirname(program_path)

    # 改变工作目录
    os.chdir(program_dir)
    print("改变工作目录 {program_dir}")

     # 启动exe文件
    print("执行exe...")
    subprocess.Popen(exe_path)
    
    print("等待窗口")
    # 等待窗口加载
    time.sleep(3)  # 根据程序启动速度调整等待时间

    # 确保窗口已经出现
    confirmed_win = confirm_window(window_title)

    # 等待额外的延时以确保应用程序已经完全加载
    time.sleep(action_delay)

    screen_shot_path = None
    screen_shot_path = screen_shot(confirmed_win, script_dir, screenshot_name, i, True)
    i+=1

    # 在应用程序中执行动作，例如点击按钮
    # 获取主窗口+截图
    app = Application(backend="uia").connect(title=window_title)
    dlg = app.window(title=window_title)

    # 找到user标签对应的文本框
    username_box = find_edit_by_label(dlg, 'ユーザー', screen_shot_path)
    username_box.set_text('SCMAINTE')

    username_box = find_edit_by_label(dlg, 'パスワード', screen_shot_path)
    username_box.set_text('0000SUPT')

    # 获取主窗口+截图
    app = Application(backend="uia").connect(title=window_title)
    dlg = app.window(title=window_title)

    # print("controls in window")
    # dlg.print_control_identifiers()

    # print("properties in control")
    # list_box = dlg.child_window(class_name="ListBox")
    # # 获取并打印ListBox对象的所有成员
    # members = inspect.getmembers(list_box)
    # for name, value in members:
    #    print(f"{name}: {value}")

    # 查找名为"xxx"的按钮并点击
    # 你可以使用按钮的名字、自动化ID、控件类型等不同属性来定位它
    print("查找按钮：開　始")
    button = dlg.child_window(title="開　始", control_type="Button")
    print("点击: 開　始")
    button.click()

    confirmed_win = confirm_window("U-PDS  Menu")
    # 等待额外的延时以确保应用程序已经完全加载
    time.sleep(action_delay)

    # 主画面截图
    screen_shot(confirmed_win, script_dir, screenshot_name, i, True)
    i+=1

    
    # 单击：人事+截图
    list_item = None
    dlg = app.window(title="U-PDS  Menu")
    list_item = dlg.child_window(title="人事", control_type="ListItem")
    print("单击项目：人事")
    list_item.click_input(coords=(5, 5))
    time.sleep(1)

    confirmed_win = confirm_window("U-PDS  Menu")
    screen_shot(confirmed_win, script_dir, screenshot_name, i, True)
    i+=1


    # 双击：给与+截图
    list_item = None
    dlg = app.window(title="U-PDS  Menu")
    list_item = dlg.child_window(title="給与", control_type="ListItem")
    # 如果控件支持 invoke 方法
    list_item.invoke()

    print("双击项目：給与")
    try:
        if list_item.exists() and list_item.is_visible():
            double_click(coords=(list_item.rectangle().mid_point()))
        else:
            print("continue")
    except TimeoutError as e:
        print("操作超时：", e)
    except ElementNotFoundError as e:
        print("未找到元素：", e)
    except Exception as e:
        print("发生错误：", e)

    confirmed_win = confirm_window("U-PDS  Menu")
    screen_shot(confirmed_win, script_dir, screenshot_name, i, True)
    i+=1

    # 双击：给与计算处理+截图
    list_item = None
    dlg = app.window(title="U-PDS  Menu")
    list_item = dlg.child_window(title="給与計算処理", control_type="ListItem")
    # 如果控件支持 invoke 方法
    list_item.invoke()

    print("給与計算処理: doucle click")
    try:
        if list_item.exists() and list_item.is_visible():
            double_click(coords=(list_item.rectangle().mid_point()))
        else:
            print("continue")
    except TimeoutError as e:
        print("操作超时：", e)
    except ElementNotFoundError as e:
        print("未找到元素：", e)
    except Exception as e:
        print("发生错误：", e)

    confirmed_win = confirm_window("U-PDS  Menu")
    screen_shot(confirmed_win, script_dir, screenshot_name, i, True)
    i+=1

    # 双击：给与计算+截图
    list_item = None
    dlg = app.window(title="U-PDS  Menu")
    list_item = dlg.child_window(title="給与計算", control_type="ListItem")
    # 如果控件支持 invoke 方法
    list_item.invoke()

    print("給与計算: doucle click")
    try:
        if list_item.exists() and list_item.is_visible():
            double_click(coords=(list_item.rectangle().mid_point()))
        else:
            print("continue")
    except TimeoutError as e:
        print("操作超时：", e)
    except ElementNotFoundError as e:
        print("未找到元素：", e)
    except Exception as e:
        print("发生错误：", e)

    confirmed_win = confirm_window("給与計算処理")
    screen_shot(confirmed_win, script_dir, screenshot_name, i, True)
    i+=1

    # 单点：利用状况照会+截图
    app = Application(backend="uia").connect(title_re="給与計算処理.*")
    dlg = app.window(title="給与計算処理")
    print("查找按钮：利用状況照会")
    button = dlg.child_window(title="利用状況照会", control_type="Button")
    print("点击：利用状況照会")
    button.click()
    
    confirmed_win = confirm_window("現在のシステム利用状況")
    screen_shot(confirmed_win, script_dir, screenshot_name, i, False)
    i+=1


    # 单点：利用状况照会+截图
    dlg = app.top_window()
    print("查找按钮：戻　る")
    button = dlg.child_window(title="戻　る", control_type="Button")
    print("点击：戻　る")
    button.click()

    confirm_window("給与計算処理")

    dlg = app.window(title="給与計算処理")
    print("查找按钮：終　了")
    button = dlg.child_window(title="終　了", control_type="Button")
    print("点击：終　了")
    button.click()

    confirmed_win = confirm_window("U-PDS  Menu")
    screen_shot(confirmed_win, script_dir, screenshot_name, i, True)
    i+=1

    app = Application(backend="uia").connect(title_re="U-PDS  Menu.*")
    dlg = app.window(title="U-PDS  Menu")
    button = dlg.child_window(title="業務終了", control_type="Button")
    print("点击：業務終了")
    button.click()

    # print("controls in window")
    # dlg = app.window(title="U-PDS  Menu")
    # dlg.print_control_identifiers()

    # print("properties in control")
    # list_box = dlg.child_window(class_name="ListBox")
    # 获取并打印ListBox对象的所有成员
    # members = inspect.getmembers(list_box)
    # for name, value in members:
    #    print(f"{name}: {value}")
    show_desktop_icons()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an EXE and capture a screenshot of its window.")
    parser.add_argument("exe_path", type=str, help="The path to the executable")
    parser.add_argument("window_title", type=str, help="The title of the window to capture")
    parser.add_argument("--action_delay", type=int, default=3, help="Time to wait before taking action")
    parser.add_argument("--screenshot_name", type=str, default="screenshot", help="Path to save the screenshot")

    args = parser.parse_args()

    runcap(args.exe_path, args.window_title, args.action_delay, args.screenshot_name)