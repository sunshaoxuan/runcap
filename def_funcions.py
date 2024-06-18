import time
import pygetwindow
import os
import pyautogui
import numpy as np
import win32gui
import win32api
import win32con
import subprocess
import sys
import tempfile
import ctypes
import pytesseract
import difflib
import pyperclip
import pywinauto
import sqlite3
import hashlib
import threading
import win32process
import re
import io
import pywinauto.keyboard as keyboard
import pywinauto.mouse as mouse
from pygetwindow import PyGetWindowException
from PIL import Image, ImageFilter, ImageEnhance, ImageChops, ImageOps
from collections import namedtuple
from pywinauto.application import Application
from pywinauto.mouse import click
from pywinauto.mouse import click as pywinauto_click
from pywinauto.mouse import double_click
from pywinauto.findwindows import ElementNotFoundError
from difflib import SequenceMatcher
from ctypes import wintypes
from pywinauto import Application, Desktop
from pywinauto.findwindows import find_windows
from pywinauto.findwindows import find_element
from pywinauto.controls.common_controls import ListViewWrapper
from pywinauto.timings import TimeoutError, wait_until_passes
from pywinauto import findwindows
from pywinauto.controls.hwndwrapper import HwndWrapper

# Constants for SendMessageTimeout
HWND_BROADCAST = 0xFFFF
WM_SETTINGCHANGE = 0x001A
SMTO_ABORTIFHUNG = 0x0002
SPI_GETDESKWALLPAPER = 0x0073
SPI_SETDESKWALLPAPER = 20
SPIF_UPDATEINIFILE = 1
SPIF_SENDCHANGE = 2
MAX_PATH = 260

IDC_ARROW = 32512
IDC_WAIT = 32514
IDC_APPSTARTING = 32650

# 自定义 HCURSOR 类型
HCURSOR = wintypes.HANDLE

WH_MOUSE_LL = 14
WM_LBUTTONDOWN = 0x0201

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("pt", POINT),
                ("mouseData", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

LowLevelMouseProc = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

hook_id = None
click_position = None
control_unique_id = None

class CURSORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("hCursor", HCURSOR),
                ("ptScreenPos", wintypes.POINT)]

user32 = ctypes.windll.user32


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def get_cursor_info():
    cursor_info = CURSORINFO()
    cursor_info.cbSize = ctypes.sizeof(CURSORINFO)
    result = user32.GetCursorInfo(ctypes.byref(cursor_info))
    if result:
        return cursor_info
    else:
        raise ctypes.WinError()

def is_mouse_busy():
    try:
        cursor_info = get_cursor_info()

        # 加载系统忙指针
        busy_cursor = user32.LoadCursorW(None, IDC_WAIT)
        app_starting_cursor = user32.LoadCursorW(None, IDC_APPSTARTING)

        # 检查当前鼠标指针是否是忙指针
        if cursor_info.hCursor == busy_cursor or cursor_info.hCursor == app_starting_cursor:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False
def init_cache_db(cache_file):
    if not cache_file or not cache_file.strip():
        return  # 如果文件名为空或仅包含空格，则忽略缓存
    with sqlite3.connect(cache_file) as conn:
        c = conn.cursor()
        # 检查表是否存在
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cache'")
        if not c.fetchone():
            # 如果表不存在，则创建表
            c.execute('''
                CREATE TABLE cache (
                    md5 TEXT PRIMARY KEY,
                    unique_id TEXT,
                    position TEXT,
                    window_id TEXT
                )
            ''')
            conn.commit()
        else:
            # 如果表已存在，检查字段是否存在
            c.execute("PRAGMA table_info(cache)")
            existing_columns = [column[1] for column in c.fetchall()]
            if 'window_id' not in existing_columns:
                # 如果 window_id 列不存在，则添加列
                c.execute('ALTER TABLE cache ADD COLUMN window_id TEXT')
                conn.commit()

# 计算MD5值
def compute_md5(query_key):
    return hashlib.md5(query_key.encode()).hexdigest()

# 缓存控件位置
def cache_control_position(cache_file, query_key, unique_id, position, window_id=None):
    init_cache_db(cache_file)
    if not cache_file or not cache_file.strip():
        return  # 如果文件名为空或仅包含空格，则忽略缓存
    md5 = compute_md5(query_key)
    with sqlite3.connect(cache_file) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO cache (md5, unique_id, position, window_id) VALUES (?, ?, ?, ?)
        ''', (md5, unique_id, position, window_id))
        conn.commit()

# 从缓存中读取控件位置
def get_cached_control_position(cache_file, query_key):
    init_cache_db(cache_file)
    if not cache_file or not cache_file.strip():
        return None  # 如果文件名为空或仅包含空格，则忽略缓存
    md5 = compute_md5(query_key)
    with sqlite3.connect(cache_file) as conn:
        c = conn.cursor()
        c.execute('''
            SELECT unique_id, position, window_id FROM cache WHERE md5 = ?
        ''', (md5,))
        result = c.fetchone()
        if result:
            return result[0], result[1], result[2]
        else:
            return None

# 清除缓存记录
def clear_cache_record(cache_file, query_key):
    init_cache_db(cache_file)
    if not cache_file or not cache_file.strip():
        return  # 如果文件名为空或仅包含空格，则忽略缓存
    md5 = compute_md5(query_key)
    with sqlite3.connect(cache_file) as conn:
        c = conn.cursor()
        c.execute('''
            DELETE FROM cache WHERE md5 = ?
        ''', (md5,))
        conn.commit()
        if c.rowcount > 0:
            print(f"Cache record for '{query_key}' cleared.")
        else:
            print(f"No cache record found for '{query_key}'.")
            
def fullwidth_to_halfwidth(text):
    result = []
    for char in text:
        unicode_code = ord(char)
        if 0xFF10 <= unicode_code <= 0xFF19:  # 全角数字0-9
            unicode_code -= 0xFF10 - ord('0')
        elif 0xFF21 <= unicode_code <= 0xFF3A:  # 全角大写字母A-Z
            unicode_code -= 0xFF21 - ord('A')
        elif 0xFF41 <= unicode_code <= 0xFF5A:  # 全角小写字母a-z
            unicode_code -= 0xFF41 - ord('a')
        result.append(chr(unicode_code))
    return ''.join(result)

def fullwidth_to_halfwidth(text):
    return ''.join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in text)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def send_input_with_api(text):
    hwnd = win32gui.GetForegroundWindow()  # 获取当前前台窗口
    for char in text:
        win32api.SendMessage(hwnd, win32con.WM_CHAR, ord(char), 0)

def send_text_via_clipboard(text):
    # 将文本复制到剪贴板
    pyperclip.copy(text)
    # 发送粘贴命令
    pyperclip.paste()

def create_white_wallpaper(width=1920, height=1080):
    # Create a white image
    image = Image.new("RGB", (width, height), "white")
    path = os.path.join(tempfile.gettempdir(), "white_wallpaper.jpg")
    image.save(path)
    return path

def set_wallpaper(wallpaper_path):
    # Create and use a white wallpaper
    wallpaper_c = ctypes.create_unicode_buffer(wallpaper_path)
    result = ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, wallpaper_c, SPIF_UPDATEINIFILE | SPIF_SENDCHANGE)
    if not result:
        error_code = ctypes.GetLastError()
        print(f"Failed to set wallpaper. Error code: {error_code}")
    else:
        print(f"Wallpaper set successfully to '{wallpaper_path}'")
        
def get_current_wallpaper():
    buffer = ctypes.create_unicode_buffer(MAX_PATH)
    ctypes.windll.user32.SystemParametersInfoW(SPI_GETDESKWALLPAPER, MAX_PATH, buffer, 0)
    return buffer.value

def save_current_settings(file_path):
    # Save the current wallpaper path
    current_wallpaper = get_current_wallpaper()
    with open(file_path, 'w') as f:
        f.write(f"{current_wallpaper}\n")
    return current_wallpaper

def restore_settings(file_path):
    # Read and set the wallpaper from the file
    with open(file_path, 'r') as f:
        wallpaper_path = f.readline().strip()
    set_wallpaper(wallpaper_path)
    return wallpaper_path

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

def find_window_by_title(title):
    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            if title in window_text:
                windows.append(hwnd)
    
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    for hwnd in windows:
        if title in win32gui.GetWindowText(hwnd):
            return hwnd
    return None

def connect_to_window_by_handle(hwnd):
    try:
        app = Application(backend="uia").connect(handle=hwnd)
        return app.window(handle=hwnd)
    except Exception as e:
        print(f"Failed to connect to window with handle '{hwnd}': {e}")
        return None

    
def send_setting_change_message():
    result = ctypes.windll.user32.SendMessageTimeoutW(
        HWND_BROADCAST,
        WM_SETTINGCHANGE,
        0,
        "Environment",
        SMTO_ABORTIFHUNG,
        5000,  # 5 seconds timeout
        ctypes.byref(ctypes.c_ulong())
    )
    if result == 0:
        error_code = ctypes.GetLastError()
        print(f"Failed to send setting change message. Error code: {error_code}")
    else:
        print("Setting change message sent successfully.")
def minimize_all_windows():
    pyautogui.hotkey('win', 'd')
    
def runcap(exe_path, check_title, time_out=10):
    # EXEファイルのパスを取得
    program_path = os.path.abspath(exe_path)

    # プログラムのディレクトリを取得
    program_dir = os.path.dirname(program_path)

    # 作業ディレクトリを変更
    os.chdir(program_dir)
    print("{program_dir}に作業ディレクトリを変更します")

     # EXEファイルを実行
    print("実行EXEファイルを起動中...")
    subprocess.Popen(exe_path)
    
    print("ウィンドウを待機中...")
    # ウィンドウが表示されるまで待機
    start_time = time.time()

    # ウィンドウが見つかるまで待機
    win = None
    while win is None:
        elapsed_time = time.time() - start_time
        if elapsed_time > time_out:
            print("タイムアウトしました：ウィンドウが見つかりませんでした。")
            # ウィンドウが見つからない場合はエラーで終了
            sys.exit(1)  

        windows = pygetwindow.getWindowsWithTitle(check_title)
        if windows:
            print(f"ウィンドウを見つけました：{check_title}")
            win = windows[0]
            # ウィンドウをアクティブにする
            win.activate()  
            break

def set_english_input():
    # 获取当前窗口句柄
    hwnd = win32gui.GetForegroundWindow()
    
    # 加载默认英文输入法
    hkl_english = win32api.LoadKeyboardLayout('00000409', win32con.KLF_ACTIVATE)

    # 设置当前窗口使用英文输入法
    win32api.SendMessage(hwnd, win32con.WM_INPUTLANGCHANGEREQUEST, 0, hkl_english)
    
def confirm_window(window_title, timeout=15):
    start_time = time.time()
    win = None
    while win is None:
        windows = pygetwindow.getWindowsWithTitle(window_title)
        if windows:
            print(f"window found: {window_title}")
            time.sleep(0.5)
            win = windows[0]

            try:
                if not win.isActive:
                    time.sleep(0.5)
                    win.activate()  # activate the window
            except PyGetWindowException as e:
                print(f"{window_title}: {e}")
        else:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout: Could not find window with title '{window_title}' within {timeout} seconds.")
    return win

def compare_image(path1, path2, output_path):
    # 加载图片并转换为灰度
    img1 = Image.open(path1).convert('L')
    img2 = Image.open(path2).convert('L')

    # 将图片转为Numpy数组
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # 计算两个数组的差异，并确定哪些像素点是不同的
    difference = np.abs(arr1.astype(int) - arr2.astype(int))
    diff = (difference > 0).astype(np.uint8)  # 将差异转化为0或1

    # 创建输出图片
    output_image = np.zeros((arr1.shape[0], arr1.shape[1], 3), dtype=np.uint8)

    # 对于灰度部分，使用原图的灰度值
    output_image[:, :, 0] = arr1 * (1 - diff)
    output_image[:, :, 1] = arr1 * (1 - diff) + 255 * diff  # 差异处标记为绿色
    output_image[:, :, 2] = arr1 * (1 - diff)

    # 将Numpy数组转回图片
    output_img = Image.fromarray(output_image, 'RGB')
    output_img.save(output_path)

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
                stable_count = 0 
        if stable_count >= wait_time:
            return current_image
            break
        prev_image = current_image
        time.sleep(0.2) 
        
def force_screen_shot(win, dir_path, file_name, suffix, adj, is_full_screen=False):
    if not is_full_screen:
        adjnum = 2
        if adj:
            adjnum = 7
        rect = win.rectangle()
        x, y, width, height = rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
        shotRegion = (x + adjnum, y + 1, width - 2 * adjnum, height - 1 - adjnum)
        print(f"Shot region: {shotRegion}")
        current_image = pyautogui.screenshot(region=shotRegion)
    else:
        current_image = pyautogui.screenshot()

    if current_image is None:
        print("Failed to capture screenshot.")
        return None
    
    screenshot_path = os.path.join(dir_path, f"{file_name}{suffix}.png")
    ensure_dir(screenshot_path)
    
    current_image.save(screenshot_path)
    print("Saved picture at: " + screenshot_path)
    return screenshot_path

def screen_shot(win, dir_path, file_name, suffix, adj, is_full_screen=False):
    if not is_full_screen:
        adjnum = 2
        if adj:
            adjnum = 7

        x, y, width, height = win.left, win.top, win.width, win.height
        shotRegion = (x + adjnum, y + 1, width - 2 * adjnum, height - 1 - adjnum)
        print(f"Shot region: {shotRegion}")
        screenshot = capture_when_stable(shotRegion)
    else:
        screenshot = capture_when_stable()

    if screenshot is None:
        print("Failed to capture screenshot.")
        return None

    screenshot_path = os.path.join(dir_path, f"{file_name}{suffix}.png")
    ensure_dir(screenshot_path)
    
    screenshot.save(screenshot_path)
    print("Saved picture at: " + screenshot_path)
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

def find_rect_by_text_ocr1(keyword, language, window_title, tesseract_path, left_offset = 0, top_offset = 0):
    # 设置 Tesseract-OCR 的路径
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    win = confirm_window(window_title)
    image_path = screen_shot(win, "tmp", "_ocr", True, False)

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

def find_rect_by_text_ocr(keyword, language, window_title, tesseract_path, min_similarity=0.5, left_offset=0, top_offset=0):
    # 设置 Tesseract-OCR 的路径
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    win = confirm_window(window_title)
    image_path = screen_shot(win, "tmp", "_ocr", True, False)
    img = Image.open(image_path).convert('L')

    def process_image(img):
        img = img.point(lambda p: p > 200 and 255)
        img = img.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
        return img

    def ocr_image(img):
        custom_config = r'--oem 3 --psm 11'
        if language:
            custom_config += f' -l {language}'
        data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)

        text_lines = {}
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                line_num = data['line_num'][i]
                if line_num not in text_lines:
                    text_lines[line_num] = []
                text_lines[line_num].append((data['text'][i], data['left'][i], data['top'][i], data['width'][i], data['height'][i]))

        return text_lines

    def find_matches(text_lines):
        potential_matches = []
        Rectangle = namedtuple('Rectangle', 'left top right bottom similarity')

        for line_num, words in text_lines.items():
            line_text = "".join(fullwidth_to_halfwidth(word[0]) for word in words)

            for start in range(len(line_text) - len(keyword) + 1):
                sub_text = line_text[start:start + len(keyword)]
                sub_similarity = SequenceMatcher(None, keyword, sub_text).ratio()
                if sub_similarity >= min_similarity:
                    weight = sum(0.01 for i in range(min(len(sub_text), len(keyword))) if sub_text[i] == keyword[i])
                    sub_similarity += weight
                    print(f"Match ratio of {sub_text} for {keyword} is : {sub_similarity}")

                    sub_start, sub_end = start, start + len(keyword)
                    x0, y0, x1, y1 = None, None, None, None
                    current_pos = 0
                    for word in words:
                        word_text = fullwidth_to_halfwidth(word[0])
                        word_length = len(word_text)
                        if current_pos <= sub_start < current_pos + word_length:
                            if x0 is None or word[1] < x0:
                                x0 = word[1]
                            if y0 is None or word[2] < y0:
                                y0 = word[2]
                        if current_pos < sub_end <= current_pos + word_length:
                            if x1 is None or (word[1] + word[3]) > x1:
                                x1 = word[1] + word[3]
                            if y1 is None or (word[2] + word[4]) > y1:
                                y1 = word[2] + word[4]
                        current_pos += word_length

                    match_rect = Rectangle(x0 + left_offset, y0 + top_offset, x1 + left_offset, y1 + top_offset, sub_similarity)
                    potential_matches.append(match_rect)

        return potential_matches

    # 正相图像处理和匹配
    processed_img = process_image(img)
    text_lines = ocr_image(processed_img)
    potential_matches = find_matches(text_lines)
    print("Processed image match finished.")

    # 反相图像处理和匹配
    inverted_img = ImageOps.invert(processed_img)
    text_lines_inverted = ocr_image(inverted_img)
    potential_matches.extend(find_matches(text_lines_inverted))
    print("Inverted image match finished.")

    # 按相似度从高到低排序
    potential_matches.sort(key=lambda rect: rect.similarity, reverse=True)

    if potential_matches:
        print(f"Found matches for '{keyword}':")
        for match in potential_matches:
            print(f"Match at position: {match} with similarity: {match.similarity}")
    else:
        print(f"'{keyword}' not found in the image.")

    return potential_matches


def find_edit_by_label(dlg_win, label_value, window_title, tesseract_path, cache_file):
    query_key = f"{window_title}_{label_value}"

    # 尝试从缓存中读取控件位置和唯一ID
    cached_result = get_cached_control_position(cache_file, query_key)
    if cached_result:
        cached_unique_id, cached_position, cached_window_id = cached_result
        print(f"Using cached position for {label_value} at {cached_position} with unique ID {cached_unique_id}")
        try:
            # 重新查找窗口并激活
            hwnd = find_window_by_title(window_title)
            if hwnd:
                print(f"Found window handle: {hwnd}")
                win32gui.SetForegroundWindow(hwnd)
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                # 使用缓存的位置进行点击操作
                x, y = map(int, cached_position.split(','))
                pywinauto_click(coords=(x, y))
                # 返回缓存的控件本身
                return dlg_win.child_window(auto_id=cached_unique_id)
        except Exception as e:
            print(f"Failed to use cached control position: {e}")

    # 如果缓存中没有找到，进行常规查找
    print(f"Attempting to find edit box with label: {label_value}")
    found_label = None
    for label in dlg_win.descendants(control_type="Edit"):
        try:
            # 尝试获取值，如果支持 ValuePattern
            current_value = label.legacy_properties()["Value"]
            if current_value == label_value:
                found_label = label
                break
        except Exception as e:
            print("此编辑框不支持 ValuePattern 或其他错误发生:", str(e))

    if found_label:
        try:
            label_rect = found_label.rectangle()
            edit_box = find_next_edit_by_rect(dlg_win, label_rect)
            if edit_box:
                edit_rect = edit_box.rectangle()
                button_position = f"{edit_rect.left},{edit_rect.top}"
                unique_id = edit_box.element_info.automation_id
                if unique_id:
                    print(f"Unique ID for caching: {unique_id}")
                    cache_control_position(cache_file, query_key, unique_id, button_position)
                    print(f"Cached position for {label_value} at {button_position}")
                return edit_box
        except Exception as e:
            print(f"Failed to cache edit box position after find: {e}")
    else:
        print(f"Label with value '{label_value}' not found. Attempting OCR.")
        # 使用 OCR 查找所有匹配
        potential_matches = find_rect_by_text_ocr(label_value, "jpn", window_title, tesseract_path)
        for match in potential_matches:
            try:
                # 将屏幕位置转换为窗口相对位置
                dlg_rect = dlg_win.rectangle()
                relative_rect = match._replace(
                    left=match.left + dlg_rect.left,
                    top=match.top + dlg_rect.top,
                    right=match.right + dlg_rect.left,
                    bottom=match.bottom + dlg_rect.top
                )
                edit_box = find_next_edit_by_rect(dlg_win, relative_rect)
                if edit_box:
                    edit_rect = edit_box.rectangle()
                    button_position = f"{edit_rect.left},{edit_rect.top}"
                    unique_id = edit_box.element_info.automation_id
                    if unique_id:
                        print(f"Unique ID for caching: {unique_id}")
                        cache_control_position(cache_file, query_key, unique_id, button_position)
                        print(f"Cached position for {label_value} at {button_position}")
                    return edit_box
            except Exception as e:
                print(f"Failed to cache edit box position after OCR: {e}")

    return None

def click_listitem(window_title, label_title, cache_file):
    query_key = f"{window_title}_{label_title}"
    
    winHwnd = None
    # 查找窗口并激活
    winHwnd = find_window_by_title(window_title)
    if winHwnd:
        print(f"Found window handle: {winHwnd}")
        win32gui.SetForegroundWindow(winHwnd)
        win32gui.ShowWindow(winHwnd, win32con.SW_RESTORE)

    dlg = connect_to_window_by_handle(winHwnd)
    allListBox = dlg.descendants(control_type="List")

    cached_result = get_cached_control_position(cache_file, query_key)
    cached_unique_id = None
    if cached_result:
        cached_unique_id, cached_position, cached_window_id = cached_result

    target_List_Item = None
    for listbox in allListBox:
            allListItems = listbox.descendants(control_type="ListItem")
            for listItem in allListItems:
                if cached_unique_id:
                    tmpUniqueId = "(" + find_after_second_comma(str(listItem.element_info.runtime_id)).strip()
                    if tmpUniqueId == cached_unique_id:
                        target_List_Item = listItem
                        print(f"Found target ListItem {label_title} from cache.")
                        break
                else:
                    item_title = listItem.window_text()
                    if item_title == label_title:
                        target_List_Item = listItem

                        # cache the list box by item title
                        control_details = get_control_details(target_List_Item)
                        unique_id = "(" + find_after_second_comma(f"{control_details['runtime_id']}").strip()
                        list_item_rect = target_List_Item.rectangle()
                        parent_position = f"{list_item_rect.left},{list_item_rect.top}"

                        if unique_id:
                            print(f"Unique ID for caching: {unique_id}")
                            cache_control_position(cache_file, query_key, unique_id, parent_position, window_title)
                            print(f"Cached position for listbox at {unique_id}")

                        print(f"Found target ListItem {label_title} from Listbox.")
                        break
            if target_List_Item:
                break

    # 找到了指定的项目并点击
    if target_List_Item:
        target_List_Item.click_input(coords=(5, 5))
        print(f"Found List Item and click it {label_title}")
        return

def get_control_details(control):
    try:
        details = {
            'automation_id': control.element_info.automation_id,
            'class_name': control.element_info.class_name,
            'control_type': control.element_info.control_type,
            'name': control.element_info.name,
            'runtime_id': control.element_info.runtime_id,
        }
    except Exception as e:
        details = {}
        print(f"Failed to get control details: {e}")
    return details

def find_after_second_comma(input_str):
    # 找到第一个逗号的位置
    first_comma_index = input_str.find(',')
    if first_comma_index == -1:
        return None  # 如果没有逗号，返回None

    # 找到第二个逗号的位置
    second_comma_index = input_str.find(',', first_comma_index + 1)
    if second_comma_index == -1:
        return None  # 如果只有一个逗号，返回None

    # 返回第二个逗号后面的所有字符
    return input_str[second_comma_index + 1:]

def double_click_listitem(window_title, label_title, cache_file):
    query_key = f"{window_title}_{label_title}"
    
    winHwnd = None
    # 查找窗口并激活
    winHwnd = find_window_by_title(window_title)
    if winHwnd:
        print(f"Found window handle: {winHwnd}")
        win32gui.SetForegroundWindow(winHwnd)
        win32gui.ShowWindow(winHwnd, win32con.SW_RESTORE)

    dlg = connect_to_window_by_handle(winHwnd)
    allListBox = dlg.descendants(control_type="List")

    cached_result = get_cached_control_position(cache_file, query_key)
    cached_unique_id = None
    if cached_result:
        cached_unique_id, cached_position, cached_window_id = cached_result

    target_List_Item = None
    for listbox in allListBox:
            allListItems = listbox.descendants(control_type="ListItem")
            for listItem in allListItems:
                if cached_unique_id:
                    tmpUniqueId = "(" + find_after_second_comma(str(listItem.element_info.runtime_id)).strip()
                    if tmpUniqueId == cached_unique_id:
                        target_List_Item = listItem
                        print(f"Found target ListItem {label_title} from cache.")
                        break
                else:
                    item_title = listItem.window_text()
                    if item_title == label_title:
                        target_List_Item = listItem

                        # cache the list box by item title
                        control_details = get_control_details(target_List_Item)
                        unique_id = "(" + find_after_second_comma(f"{control_details['runtime_id']}").strip()
                        list_item_rect = target_List_Item.rectangle()
                        parent_position = f"{list_item_rect.left},{list_item_rect.top}"

                        if unique_id:
                            print(f"Unique ID for caching: {unique_id}")
                            cache_control_position(cache_file, query_key, unique_id, parent_position, window_title)
                            print(f"Cached position for listbox at {unique_id}")

                        print(f"Found target ListItem {label_title} from Listbox.")
                        break
            if target_List_Item:
                break

    # 找到了指定的项目并点击
    if target_List_Item:
        target_List_Item.invoke()
        print(f"Found List Item and double click it {label_title}")
        return

    try:
        if target_List_Item.exists() and target_List_Item.is_visible():
            pywinauto.mouse.double_click(coords=(target_List_Item.rectangle().mid_point()))
    except TimeoutError as e:
        print("Timeout: ", e)
    except ElementNotFoundError as e:
        print("Element not found: ", e)
    except Exception as e:
        print("Error appeared: ", e)

        
def find_app_by_automation_id(automation_id):
    try:
        print(f"Trying to find application with AutomationId '{automation_id}'")

        windows = Desktop(backend="uia").windows()
        for win in windows:
            current_automation_id = win.automation_id()  # 正确获取 AutomationId
            window_text = win.window_text()
            print(f"Found window: '{window_text}', AutomationId: '{current_automation_id}'")
            if current_automation_id == automation_id:
                print(f"Exact match found: '{window_text}' with AutomationId: '{automation_id}'")
                return win

        print(f"No exact match found for AutomationId '{automation_id}'")
        return None
    except Exception as e:
        print(f"Failed to find app by AutomationId '{automation_id}': {e}")
        return None
    
def find_button_by_title_and_class(parent_hwnd, title, class_name):
    def enum_child_windows_callback(hwnd, buttons):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            if title in window_text and class_name == "Button":
                buttons.append(hwnd)
    
    buttons = []
    win32gui.EnumChildWindows(parent_hwnd, enum_child_windows_callback, buttons)
    
    if buttons:
        return buttons[0]
    return None

def activate_window_by_handle(window_handle):
    try:
        app = Application().connect(handle=window_handle)
        win = app.window(handle=window_handle)
        win.set_focus()  # 确保窗口被激活
    except Exception as e:
        print(f"Error activating window with handle {window_handle}: {e}")


def click_button(window_title, label_title, cache_file, automation_id=None):
    query_key = f"{window_title}_{label_title}"
    
    # 初始化缓存数据库
    init_cache_db(cache_file)
    
    # 尝试从缓存中读取控件位置和窗口唯一ID
    cached_result = get_cached_control_position(cache_file, query_key)
    if cached_result:
        cached_unique_id, cached_position, cached_window_id = cached_result
        print(f"Using cached position for {label_title} at {cached_position} with window id {cached_window_id}")
        
        # 重新查找窗口并激活
        hwnd = find_window_by_title(window_title)
        if hwnd:
            print(f"Found window handle: {hwnd}")
            win32gui.SetForegroundWindow(hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            # 使用缓存的位置进行点击操作
            x, y = map(int, cached_position.split(','))
            pywinauto_click(coords=(x, y))
            print("Button clicked using cached position.")
            return

    # 如果缓存中没有找到，进行常规查找
    try:
        if automation_id:
            print(f"Attempting to find application with AutomationId: {automation_id}")
            win = find_app_by_automation_id(automation_id)
        else:
            print(f"Attempting to find application with title: {window_title}")
            hwnd = find_window_by_title(window_title)
            if hwnd:
                print(f"Found window handle: {hwnd}")
                # 激活窗口
                win32gui.SetForegroundWindow(hwnd)
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win = connect_to_window_by_handle(hwnd)
            else:
                win = None
        
        if win is None:
            print(f"Application with title '{window_title}' or AutomationId '{automation_id}' not found.")
            return
        
        print(f"Attempting to find window with title: {window_title}")
        dlg = win

        # 等待窗口完全加载
        print("Waiting for window to be ready...")
        wait_until_passes(20, 0.5, lambda: dlg.exists(), exceptions=(TimeoutError,))

        # 使用 Win32 API 查找按钮
        print(f"Attempting to find button with title: {label_title} using Win32 API")
        button_hwnd = find_button_by_title_and_class(hwnd, label_title, "Button")
        if not button_hwnd:
            print(f"Button with title '{label_title}' not found using Win32 API.")
            return
        
        # 获取按钮的坐标
        rect = win32gui.GetWindowRect(button_hwnd)
        x, y = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
        
        # 使用 pywinauto 模拟鼠标点击
        print("Clicking the button...")
        pywinauto_click(coords=(x, y))
        print("Button clicked.")

        # 尝试获取控件的位置和唯一ID并缓存起来
        try:
            button_position = f"{rect[0]},{rect[1]}"
            unique_id = str(button_hwnd)
            window_id = automation_id if automation_id else str(hwnd)
            if unique_id and window_id:
                print(f"Unique ID for caching: {unique_id}, Window ID for caching: {window_id}")
                cache_control_position(cache_file, query_key, unique_id, button_position, window_id)
                print(f"Cached position for {label_title} at {button_position}")
            else:
                print(f"No unique ID or window ID found for button '{label_title}'.")
        except Exception as e:
            print(f"Failed to cache button position after click: {e}")
    
    except TimeoutError as e:
        print(f"Failed to find or interact with the button: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def set_foreground_window(hwnd):
    """设置窗口到前台"""
    try:
        win32api.AllowSetForegroundWindow(win32con.ASFW_ANY)
        win32gui.SetForegroundWindow(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    except Exception as e:
        print(f"Error setting foreground window: {e}")

def modal_table_click(window_title, modal_dialog_title, key_word, tesseract_path, click_type, cache_file):
    query_key_dialog = f"{window_title}_{modal_dialog_title}"
    query_key_control = f"{window_title}_{modal_dialog_title}_{key_word}"

    # 尝试从缓存中读取对话框位置和唯一ID
    cached_dialog_result = get_cached_control_position(cache_file, query_key_dialog)
    if cached_dialog_result:
        cached_dialog_unique_id, cached_dialog_position, _ = cached_dialog_result
        print(f"Using cached position for dialog {modal_dialog_title} at {cached_dialog_position} with unique ID {cached_dialog_unique_id}")
        hwnd = find_window_by_title(window_title)
        if hwnd:
            print(f"Found window handle: {hwnd}")
            set_foreground_window(hwnd)
            dialog_x, dialog_y = map(int, cached_dialog_position.split(','))
            pywinauto.mouse.click(coords=(dialog_x, dialog_y))
            if not is_mouse_busy():
                dlg = connect_to_window_by_handle(hwnd)
                dialog = dlg.child_window(auto_id=cached_dialog_unique_id)
            else:
                dialog = None
        else:
            dialog = None
    else:
        dialog = None

    if dialog is None:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

        hwnd = find_window_by_title(modal_dialog_title)
        win32gui.SetForegroundWindow(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        dialog = connect_to_window_by_handle(hwnd)

        if dialog is not None:
            dialog.set_focus()
            rect = dialog.rectangle()
            rect = dialog.rectangle()
            center_x, center_y = rect.mid_point().x, rect.mid_point().y
            mouse.click(coords=(center_x, center_y))
            unique_id = dialog.element_info.automation_id
            if unique_id:
                print(f"Unique ID for dialog caching: {unique_id}")
                cache_control_position(cache_file, query_key_dialog, unique_id, dialog_position)
                print(f"Cached position for dialog {modal_dialog_title} at {dialog_position}")
        else:
            app = Application(backend="uia").connect(title=window_title)
            base_window = app.window(title=window_title)

            for dialog in base_window.descendants():
                if dialog.element_info.name == modal_dialog_title and dialog.class_name() == "Gupta:Dialog":
                    dialog.set_focus()
                    rect = dialog.rectangle()
                    center_x, center_y = rect.mid_point().x, rect.mid_point().y

                    pywinauto.mouse.click(coords=(center_x, center_y))
                    if not is_mouse_busy():
                        dialog_position = f"{center_x},{center_y}"
                        unique_id = dialog.element_info.automation_id
                        if unique_id:
                            print(f"Unique ID for dialog caching: {unique_id}")
                            cache_control_position(cache_file, query_key_dialog, unique_id, dialog_position)
                            print(f"Cached position for dialog {modal_dialog_title} at {dialog_position}")
                        break
            else:
                print("Dialog not found.")
                return

        while not dialog.is_active():
            time.sleep(0.5)

    # 获取对话框的屏幕坐标
    dialog_rect = dialog.rectangle()

    # 尝试从缓存中读取控件位置和唯一ID
    cached_result = get_cached_control_position(cache_file, query_key_control)
    print(f"cached result info: {cached_result}")
    if cached_result:
        cached_unique_id, cached_position, _ = cached_result
        print(f"Using cached position for {key_word} at {cached_position} with unique ID {cached_unique_id}")
        try:
            # 使用缓存的位置进行点击或双击操作
            x, y = map(int, cached_position.split(','))
            if click_type == 0:
                pywinauto.mouse.click(coords=(x, y))
            else:
                pywinauto.mouse.double_click(coords=(x, y))
            print(f"{'Double click' if click_type else 'Click'} performed at:", (x, y))
            return
        except Exception as e:
            print(f"Failed to use cached control position: {e}")

    potential_matches = find_rect_by_text_ocr(key_word, "jpn", modal_dialog_title, pytesseract.pytesseract.tesseract_cmd)
    for rect in potential_matches:
        try:
            # 根据对话框的屏幕位置调整坐标
            mid_x = rect.left + (rect.right - rect.left) // 2 + dialog_rect.left
            mid_y = rect.top + (rect.bottom - rect.top) // 2 + dialog_rect.top

            if click_type == 0:
                pywinauto.mouse.click(coords=(mid_x, mid_y))
            else:
                pywinauto.mouse.double_click(coords=(mid_x, mid_y))
            print(f"{'Double click' if click_type else 'Click'} performed at:", (mid_x, mid_y))

            # 缓存找到的控件位置和唯一ID
            item_position = f"{mid_x},{mid_y}"
            unique_id = rect.left  # 使用找到的矩形左坐标作为唯一ID的替代
            if unique_id:
                print(f"Unique ID for caching: {unique_id}")
                cache_control_position(cache_file, query_key_control, unique_id, item_position)
                print(f"Cached position for {key_word} at {item_position}")
            return
        except Exception as e:
            print(f"Failed to click and cache position: {e}")

    print("Target cell not found.")

def window_table_click(window_title, key_word, click_type, tesseract_path, cache_file):
    pytesseract.pytesseract.cmd = tesseract_path

    # 查找窗口句柄并激活窗口
    hwnd = find_window_by_title(window_title)
    if hwnd:
        print(f"Found window handle: {hwnd}")
        set_foreground_window(hwnd)
        dlg = connect_to_window_by_handle(hwnd)
    else:
        print(f"Window with title '{window_title}' not found.")
        exit(1)

    query_key = f"{window_title}_{key_word}"

    # 尝试从缓存中读取控件位置和唯一ID
    cached_result = get_cached_control_position(cache_file, query_key)
    if cached_result:
        cached_unique_id, cached_position, _ = cached_result
        print(f"Using cached position for {key_word} at {cached_position} with unique ID {cached_unique_id}")
        try:
            # 使用缓存的位置进行点击或双击操作
            x, y = map(int, cached_position.split(','))
            if click_type == 0:
                pywinauto.mouse.click(coords=(x, y))
            else:
                pywinauto.mouse.double_click(coords=(x, y))
            print(f"{'Double click' if click_type else 'Click'} performed at:", (x, y))
            return
        except Exception as e:
            print(f"Failed to use cached control position: {e}")

    # 获取窗口坐标
    base_window = dlg
    base_window.set_focus()
    rect_win = base_window.rectangle()
    center_x, center_y = rect_win.mid_point().x, rect_win.mid_point().y

    # 在窗口中心点进行单击，以取消任何选中的行
    mouse.click(coords=(center_x, center_y))

    potential_matches = find_rect_by_text_ocr(key_word, "jpn", window_title, pytesseract.pytesseract.cmd)
    rect = potential_matches[0]

    item_position = None

    try:
        # 根据对话框的屏幕位置调整坐标
        mid_x = rect.left + (rect.right - rect.left) // 2 + rect_win.left
        mid_y = rect.top + (rect.bottom - rect.top) // 2 + rect_win.top

        # 启动线程设置鼠标钩子并执行点击操作
        hook_thread = threading.Thread(target=set_mouse_hook_and_click, args=(mid_x, mid_y, click_type, hwnd))
        hook_thread.start()
        hook_thread.join(timeout=2)  # 设置超时时间，防止无限等待


        # 添加以下代码调用 get_control_unique_id_at_position
        control_unique_id = get_control_unique_id_at_position(mid_x, mid_y, hwnd)
        if control_unique_id:
            print(f"Unique ID for caching: {control_unique_id}")
            cache_control_position(cache_file, query_key, control_unique_id, item_position)
            print(f"Cached position for {key_word} at {item_position}")
        else:
            print("Failed to capture control unique ID.")
        
        # 缓存找到的控件位置和唯一ID
        item_position = f"{mid_x},{mid_y}"
        if control_unique_id:
            print(f"Unique ID for caching: {control_unique_id}")
            cache_control_position(cache_file, query_key, control_unique_id, item_position)
            print(f"Cached position for {key_word} at {item_position}")
        else:
            print("Failed to capture control unique ID.")
        return
    except Exception as e:
        print(f"Failed to click and cache position: {e}")

def get_control_unique_id_at_position(x, y, hwnd):
    def enum_child_windows_callback(child_hwnd, param):
        child_rect = win32gui.GetWindowRect(child_hwnd)
        child_x0, child_y0, child_x1, child_y1 = child_rect
        if child_x0 <= x <= child_x1 and child_y0 <= y <= child_y1:
            param.append(child_hwnd)
        return True

    matching_hwnds = []
    win32gui.EnumChildWindows(hwnd, enum_child_windows_callback, matching_hwnds)
    if matching_hwnds:
        matching_hwnd = matching_hwnds[0]
        try:
            element = pywinauto.findwindows.find_element(handle=matching_hwnd)
            return element.control_id
        except Exception as e:
            print(f"Failed to get control unique ID at position ({x}, {y}): {e}")
            return None
    else:
        print(f"No matching control found at position ({x}, {y})")
        return None

def mouse_proc(nCode, wParam, lParam):
    global click_position, control_unique_id
    if wParam == WM_LBUTTONDOWN:
        hook_struct = ctypes.cast(lParam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
        x, y = hook_struct.pt.x, hook_struct.pt.y
        print(f"Mouse clicked at ({x}, {y})")
        click_position = (x, y)
        control_unique_id = get_control_unique_id_at_position(x, y)
        ctypes.windll.user32.UnhookWindowsHookEx(hook_id)
        return 1
    return ctypes.windll.user32.CallNextHookEx(hook_id, nCode, wParam, lParam)

def set_mouse_hook():
    global hook_id
    hook_id = ctypes.windll.user32.SetWindowsHookExW(WH_MOUSE_LL, LowLevelMouseProc(mouse_proc), ctypes.windll.kernel32.GetModuleHandleW(None), 0)

def remove_mouse_hook():
    global hook_id
    if hook_id:
        ctypes.windll.user32.UnhookWindowsHookEx(hook_id)
        hook_id = None

def set_mouse_hook_and_click(mid_x, mid_y, click_type, hwnd):
    global control_unique_id
    set_mouse_hook()

    # 点击或双击操作
    if click_type == 0:
        pywinauto.mouse.click(coords=(mid_x, mid_y))
    else:
        pywinauto.mouse.double_click(coords=(mid_x, mid_y))
    print(f"{'Double click' if click_type else 'Click'} performed at:", (mid_x, mid_y))

    # 调用 get_control_unique_id_at_position 并传递 hwnd
    control_unique_id = get_control_unique_id_at_position(mid_x, mid_y, hwnd)

    # 等待鼠标钩子捕获点击事件
    time.sleep(2)  # 等待2秒钟，确保捕获到点击事件

    remove_mouse_hook()

def edit_from_window_table(window_title, y_title, x_title, key_word, tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    app = Application(backend="uia").connect(title=window_title)
    base_window = app.window(title=window_title)
    
    base_window.set_focus()
    rect_win = base_window.rectangle()
    # center_x, center_y = rect_win.mid_point().x, rect_win.mid_point().y

    # 在窗口中心点进行单击，以取消任何选中的行
    # mouse.click(coords=(center_x, center_y))
    # time.sleep(1)  # 等待界面反应

    # 找到item_title的位置
    rect_item = find_rect_by_text_ocr(y_title, "jpn", window_title, pytesseract.pytesseract.tesseract_cmd)
    # 找到x_title的位置
    rect_search = find_rect_by_text_ocr(x_title, "jpn", window_title, pytesseract.pytesseract.tesseract_cmd, 0.5, 0, 0, rect_item.left + rect_item.right)

    if rect_item and rect_search:
        mid_x = rect_search[2] + rect_win.left
        mid_y = (rect_item[1] + rect_item[3]) // 2 + rect_win.top
        pywinauto.mouse.click(coords=(mid_x, mid_y))
        pyperclip.copy(key_word)  # 将文本复制到剪贴板
        pyautogui.hotkey('ctrl', 'v')  # 粘贴文本
        time.sleep(0.5)  # 等待输入法处理
        pyautogui.press('enter')  # 按下回车键确认输入
        print("Text {} entered at: {}, {}".format(key_word, mid_x, mid_y))
    else:
        if not rect_item:
            print("Item cell not found.")
        if not rect_search:
            print("Search condition cell not found.")

def key_word_click(window_title, key_word, tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    app = Application(backend="uia").connect(title=window_title)
    base_window = app.window(title=window_title)
    
    base_window.set_focus()
    rect_win = base_window.rectangle()
    # center_x, center_y = rect_win.mid_point().x, rect_win.mid_point().y

    # 在窗口中心点进行单击，以取消任何选中的行
    # mouse.click(coords=(center_x, center_y))
    # time.sleep(1)  # 等待界面反应

    # 找到item_title的位置
    rect_item = find_rect_by_text_ocr(key_word, "jpn", window_title, pytesseract.pytesseract.tesseract_cmd)
    # 找到x_title的位置
    rect_search = find_rect_by_text_ocr(key_word, "jpn", window_title, pytesseract.pytesseract.tesseract_cmd, 0.5, 0, 0, rect_item.left + rect_item.right)

    if rect_item and rect_search:
        mid_x = rect_search[2] + rect_win.left
        mid_y = (rect_item[1] + rect_item[3]) // 2 + rect_win.top
        pywinauto.mouse.click(coords=(mid_x, mid_y))
        print("Click at: {}, {}".format(mid_x, mid_y))
    else:
        if not rect_item:
            print("Item cell not found.")
        if not rect_search:
            print("Search condition cell not found.")

def get_window_by_handle(handle):
    try:
        # 获取应用程序对象
        app = Application().connect(handle=handle)
        # 获取窗口对象
        win = app.window(handle=handle)
        return win
    except Exception as e:
        print(f"Failed to get window or app by handle: {e}")
        return None
    
def click_list_item_by_handle(list_item_handle, double_click=False):
    try:
        # 连接到应用程序
        app = Application().connect(handle=list_item_handle)
        
        # 获取 ListItem 对象
        list_item_element = find_element(handle=list_item_handle)
        
        # 获取 ListView 父控件
        list_view_handle = list_item_element.rectangle().left
        list_view = ListViewWrapper(list_view_handle)
        
        # 获取项的矩形
        item_rect = list_item_element.rectangle()
        
        # 将鼠标移动到项目的中心
        item_center = (item_rect.left + item_rect.width() // 2, item_rect.top + item_rect.height() // 2)
        
        # 单击或双击项目
        if double_click:
            pywinauto.mouse.double_click(coords=item_center)
        else:
            pywinauto.mouse.click(coords=item_center)
        
        print(f"{'Double clicked' if double_click else 'Clicked'} list item with handle: {list_item_handle}")
    
    except Exception as e:
        print(f"Failed to interact with the list item: {e}")