from def_funcions import confirm_window, screen_shot
import argparse
import def_funcions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ウィンドウが安定したときにスクリーンショットを取ります")
    parser.add_argument("window_title", type=str, help="実行中のプログラムのウィンドウのタイトル")
    parser.add_argument("dir_path", type=str, help="スクリーンショットのパス")
    parser.add_argument("file_name", type=str, help="スクリーンショットのファイル名")
    parser.add_argument("suffix", type=str, default="", help="ファイル名のサフィックス（デフォルトは空です）")
    args = parser.parse_args()
    
    try:
        confirmed_win = confirm_window(args.window_title)
        screen_shot(confirmed_win, args.dir_path, args.file_name, args.suffix, adj=True, is_full_screen=False)
    except Exception as e:
        raise e;