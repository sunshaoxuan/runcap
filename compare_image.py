import argparse
import def_funcions
from def_funcions import compare_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像を比較して違いを強調する")
    parser.add_argument("path1", type=str, help="比較する製品１の画像ファイルのパス")
    parser.add_argument("path2", type=str, help="比較する製品２の画像ファイルのパス")
    parser.add_argument("output_path", type=str, help="差分画像を保存するパス")

    args = parser.parse_args()
    
    compare_image(args.path1, args.path2, args.output_path)
