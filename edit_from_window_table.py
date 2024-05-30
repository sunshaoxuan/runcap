import argparse
import def_funcions
from def_funcions import edit_from_window_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ウィンドウのテーブル項目にキーワードを入力する")
    parser.add_argument("window_title", type=str, help="実行中のプログラムのウィンドウのタイトル")
    parser.add_argument("item_title", type=str, help="入力するプロジェクトの行のタイトル")
    parser.add_argument("head_title", type=str, help="入力するプロジェクトの列のタイトル")
    parser.add_argument("key_word", type=str, help="検索するキーワード")
    parser.add_argument("tesseract_path", type=str, default="C:\\Program Files\\Tesseract-OCR\\tesseract.exe", help="Tesseractのパス")
    parser.add_argument("--time_out", type=int, default=10, help="待機するタイムアウト時間（デフォルトは10秒です）")

    
    args = parser.parse_args()
    edit_from_window_table(args.window_title, args.item_title, args.head_title, args.key_word, args.tesseract_path)
