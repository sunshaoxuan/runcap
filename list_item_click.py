import argparse
import def_funcions
from def_funcions import click_listitem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="リストアイテムをクリックする")
    parser.add_argument("window_title", type=str, help="実行中のプログラムのウィンドウのタイトル")
    parser.add_argument("item_title", type=str, help="リスト項目のテキスト")
    parser.add_argument("cache_file", type=str, help="キャッシュファイルのパス")

    args = parser.parse_args()
    click_listitem(args.window_title, args.item_title, args.cache_file)
