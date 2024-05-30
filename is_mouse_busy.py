import def_funcions
from def_funcions import is_mouse_busy

if __name__ == "__main__":
    if is_mouse_busy():
        print("Mouse is busy")
    else:
        print("Mouse is idle")