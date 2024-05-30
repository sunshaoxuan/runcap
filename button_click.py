import argparse
import def_funcions
from def_funcions import click_button

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ボタンをクリックする")
    parser.add_argument("window_title", type=str, help="実行中のプログラムのウィンドウのタイトル")
    parser.add_argument("label_title", type=str, help="ボタンのタイトル")
    parser.add_argument("cache_file", type=str, help="キャッシュファイルのパス")
    
    args = parser.parse_args()
    click_button(args.window_title, args.label_title, args.cache_file)
