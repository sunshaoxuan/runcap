import def_funcions
import argparse
from def_funcions import toggle_desktop_icons

def str_to_bool(value):
    if value.lower() in ('yes', 'true', 't', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="デスクトップアイコンを表示しますか。")
    parser.add_argument("isshown", type=str_to_bool, help="true: 表示する、false: 表示しない")

    args = parser.parse_args()
    
    toggle_desktop_icons(args.isshown)
