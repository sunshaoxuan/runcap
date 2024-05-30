import argparse
import def_funcions
from def_funcions import modal_table_click

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指定されたモーダルダイアログの特定のテーブルテキストをダブルクリックする")
    parser.add_argument("window_title", type=str, help="実行中のプログラムのウィンドウのタイトル")
    parser.add_argument("modal_dialog_title", type=str, help="指定されたモーダルダイアログ")
    parser.add_argument("key_word", type=str, help="検索するキーワード")
    parser.add_argument("click_type", type=int, help="クリックのタイプ（０：クリック、１：ダブルクリック）")
    parser.add_argument("tesseract_path", type=str, default="C:\\Program Files\\Tesseract-OCR\\tesseract.exe", help="Tesseractのパス")
    parser.add_argument("cache_file", type=str, help="キャッシュファイルのパス")
    parser.add_argument("--time_out", type=int, default=10, help="待機するタイムアウト時間（デフォルトは10秒です）")
    
    args = parser.parse_args()
    modal_table_click(args.window_title, args.modal_dialog_title, args.key_word, args.tesseract_path, args.click_type, args.cache_file)
