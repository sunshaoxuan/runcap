import def_funcions
import argparse
from def_funcions import save_current_settings, create_white_wallpaper, set_wallpaper, restore_settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set or restore the desktop background to white")
    parser.add_argument("action", type=str, choices=['set', 'restore'], help="Choose 'set' to set a white background, 'restore' to restore the original background")
    parser.add_argument("--file_path", type=str, default="background_settings.txt", help="File path to save or restore the wallpaper path")
    
    args = parser.parse_args()
    
    if args.action == 'set':
        wallpaper_path = save_current_settings(args.file_path)
        print(f"Wallpaper save successfully to '{wallpaper_path}'")
        white_wallpaper_path = create_white_wallpaper()
        print(f"White wallpaper save successfully to '{white_wallpaper_path}'")
        set_wallpaper(white_wallpaper_path)
    elif args.action == 'restore':
        restore_settings(args.file_path)
