from pywinauto.application import Application
import argparse
from def_funcions import find_edit_by_label, find_window_by_title, connect_to_window_by_handle
import def_funcions
import win32gui
import win32con

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="指定タイトルの右側のテキストボックスに文字を入力する")
    parser.add_argument("window_title", type=str, help="実行中のウィンドウのタイトル")
    parser.add_argument("textbox_label", type=str, help="テキストボックスの左側のタイトル")
    parser.add_argument("input_value", type=str, help="入力する文字")
    parser.add_argument("tesseract_path", type=str, default="", help="Tesseractのパス")
    parser.add_argument("cache_file", type=str, help="キャッシュファイルのパス")
    parser.add_argument("--time_out", type=int, default=10, help="待機するタイムアウト時間（デフォルトは10秒です）")
    args = parser.parse_args()

    # 查找窗口句柄并激活窗口
    hwnd = find_window_by_title(args.window_title)
    if hwnd:
        print(f"Found window handle: {hwnd}")
        win32gui.SetForegroundWindow(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        dlg = connect_to_window_by_handle(hwnd)
    else:
        print(f"Window with title '{args.window_title}' not found.")
        exit(1)

    inputbox = find_edit_by_label(dlg, args.textbox_label, args.window_title, args.tesseract_path, args.cache_file)

    if inputbox:
        inputbox.set_text(args.input_value)
    else:
        print("Edit box not found.")