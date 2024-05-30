import argparse
import def_funcions
from def_funcions import runcap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EXEファイルを実行する")
    parser.add_argument("exe_path", type=str, help="実行可能ファイルへのパス")
    parser.add_argument("check_title", type=str, help="実行中のプログラムのウィンドウのタイトル")
    parser.add_argument("--time_out", type=int, default=10, help="待機するタイムアウト時間（デフォルトは10秒です）")
    args = parser.parse_args()
    
    runcap(args.exe_path, args.check_title, args.time_out)
