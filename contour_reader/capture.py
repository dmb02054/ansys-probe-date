"""화면 캡처 + 드래그 영역 선택 오버레이."""
from __future__ import annotations

import ctypes
import tkinter as tk

from PIL import Image, ImageEnhance, ImageGrab, ImageTk


def enable_dpi_awareness() -> None:
    """Tk 좌표와 실제 화면 픽셀을 일치시킨다 (Tk 생성 전에 호출)."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _virtual_screen() -> tuple[int, int, int, int]:
    u = ctypes.windll.user32
    return u.GetSystemMetrics(76), u.GetSystemMetrics(77), u.GetSystemMetrics(78), u.GetSystemMetrics(79)


def grab_virtual_screen() -> tuple[Image.Image, int, int]:
    left, top, _, _ = _virtual_screen()
    return ImageGrab.grab(all_screens=True).convert("RGB"), left, top


def select_region(root: tk.Misc) -> Image.Image | None:
    """전체 화면을 캡처해 오버레이로 띄우고, 드래그한 영역 이미지를 반환 (ESC/우클릭 취소)."""
    shot, left, top = grab_virtual_screen()
    w, h = shot.size
    dim = ImageEnhance.Brightness(shot).enhance(0.45)

    win = tk.Toplevel(root)
    win.overrideredirect(True)
    win.geometry(f"{w}x{h}{left:+d}{top:+d}")
    win.attributes("-topmost", True)
    canvas = tk.Canvas(win, width=w, height=h, highlightthickness=0, cursor="crosshair")
    canvas.pack()
    bg = ImageTk.PhotoImage(dim)
    canvas.create_image(0, 0, anchor="nw", image=bg)
    primary_w = ctypes.windll.user32.GetSystemMetrics(0)
    canvas.create_text(-left + primary_w // 2, -top + 40,
                       text="캡처할 영역을 드래그하세요  (ESC / 우클릭: 취소)",
                       fill="white", font=("Malgun Gothic", 16, "bold"))

    state = {"start": None, "rect": None, "label": None, "result": None}

    def press(e):
        state["start"] = (e.x, e.y)

    def drag(e):
        if not state["start"]:
            return
        x0, y0 = state["start"]
        canvas.delete("sel")
        canvas.create_rectangle(x0, y0, e.x, e.y, outline="#00e5ff", width=2, tags="sel")
        canvas.create_text(max(x0, e.x), max(y0, e.y) + 14, anchor="ne", fill="#00e5ff",
                           font=("Malgun Gothic", 11, "bold"),
                           text=f"{abs(e.x - x0)} x {abs(e.y - y0)}", tags="sel")

    def release(e):
        if not state["start"]:
            return
        x0, y0 = state["start"]
        x0, x1 = sorted((max(0, x0), min(w, e.x)))
        y0, y1 = sorted((max(0, y0), min(h, e.y)))
        if x1 - x0 >= 5 and y1 - y0 >= 5:
            state["result"] = shot.crop((x0, y0, x1, y1))
            win.destroy()
        else:
            state["start"] = None
            canvas.delete("sel")

    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", release)
    canvas.bind("<ButtonPress-3>", lambda e: win.destroy())
    win.bind("<Escape>", lambda e: win.destroy())
    win.update()
    win.focus_force()
    canvas.focus_set()
    root.wait_window(win)
    return state["result"]
