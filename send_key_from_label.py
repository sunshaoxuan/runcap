from pywinauto.application import Application
import argparse
import def_funcions
from def_funcions import find_edit_by_label, find_app_by_title

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="指定されたタイトルの右側にあるテキストボックスで指定キーを送信する")
    parser.add_argument("window_title", type=str, help="実行中のウィンドウのタイトル")
    parser.add_argument("textbox_label", type=str, help="テキストボックスの左側のタイトル")
    parser.add_argument("key_value", type=str, help="送信するキーボードのキー値")
    parser.add_argument("tesseract_path", type=str, default="", help="Tesseractのパス")
    parser.add_argument("--time_out", type=int, default=10, help="待機するタイムアウト時間（デフォルトは10秒です）")
    args = parser.parse_args()
    
    app =  find_app_by_title(args.window_title)
    dlg = app.window(title=args.window_title)
    inputbox = find_edit_by_label(dlg, args.textbox_label, args.window_title, args.tesseract_path)

    if inputbox:       
        inputbox.type_keys(args.key_value)
    else:
        print("Edit box not found.")
