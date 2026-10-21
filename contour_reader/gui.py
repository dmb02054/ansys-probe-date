"""컨투어 이미지 값 판독기 GUI."""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageGrab, ImageTk

from . import capture, settings
from .legend import (DEFAULT_COLORS, Legend, Probe, fmt, hex_to_rgb,
                     resample_colors, sample_colors)

FONT = "Malgun Gothic"


def _text_color(hex_color: str) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return "#000000" if (r * 299 + g * 587 + b * 114) / 1000 > 140 else "#ffffff"


class App(tk.Tk):
    def __init__(self):
        capture.enable_dpi_awareness()
        super().__init__()
        self.title("컨투어 값 판독기")
        self.geometry("1280x860")
        self.minsize(900, 640)

        cfg = settings.load()
        self.colors: list[str] = list(cfg["colors"])
        self.v_min = tk.StringVar(value=str(cfg["vmin"]))
        self.v_max = tk.StringVar(value=str(cfg["vmax"]))
        self.v_unit = tk.StringVar(value=cfg["unit"])
        self.v_n = tk.IntVar(value=len(self.colors))
        self.v_tol = tk.IntVar(value=int(cfg["tolerance"]))
        self.v_sample = tk.IntVar(value=int(cfg["sample"]))
        self.v_open = tk.BooleanVar(value=bool(cfg["open_ends"]))
        self.v_delay = tk.IntVar(value=int(cfg["delay"]))
        self.v_hover = tk.BooleanVar(value=bool(cfg["hover"]))

        self.img: Image.Image | None = None
        self.arr: np.ndarray | None = None
        self.photo = None
        self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0
        self.markers: list[tuple[int, int]] = []
        self.legend_mode = False
        self._press = None
        self._drag_rect = None

        self._build_ui()
        self._bind_keys()
        for v in (self.v_min, self.v_max, self.v_unit, self.v_tol, self.v_sample, self.v_open):
            v.trace_add("write", lambda *_: self._on_legend_change())
        self._refresh_swatches()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)

        # 이미지 입력
        f = ttk.LabelFrame(left, text="1. 이미지 가져오기", padding=6)
        f.pack(fill="x")
        ttk.Button(f, text="화면 영역 캡처 (F2)", command=self.start_capture).grid(
            row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(f, text="캡처 지연(초)").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Spinbox(f, from_=0, to=10, width=5, textvariable=self.v_delay).grid(row=1, column=1, sticky="e")
        ttk.Button(f, text="파일 열기 (Ctrl+O)", command=self.open_file).grid(row=2, column=0, sticky="ew")
        ttk.Button(f, text="클립보드 붙여넣기 (Ctrl+V)", command=self.paste).grid(row=2, column=1, sticky="ew")
        f.columnconfigure((0, 1), weight=1)

        # 범례 값
        f = ttk.LabelFrame(left, text="2. 범례 값 입력", padding=6)
        f.pack(fill="x", pady=6)
        ttk.Label(f, text="최소값").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.v_min, width=10).grid(row=0, column=1, padx=4)
        ttk.Label(f, text="최대값").grid(row=0, column=2, sticky="w")
        ttk.Entry(f, textvariable=self.v_max, width=10).grid(row=0, column=3, padx=4)
        ttk.Label(f, text="단위").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(f, textvariable=self.v_unit, width=10).grid(row=1, column=1, padx=4)
        ttk.Label(f, text="구간 수").grid(row=1, column=2, sticky="w")
        ttk.Spinbox(f, from_=2, to=30, width=8, textvariable=self.v_n,
                    command=self._on_n_change).grid(row=1, column=3, padx=4)
        ttk.Checkbutton(f, text="양 끝 구간을 '이상/이하'로 표시", variable=self.v_open).grid(
            row=2, column=0, columnspan=4, sticky="w")

        # 범례 색상
        f = ttk.LabelFrame(left, text="3. 범례 색상 (위=최대)", padding=6)
        f.pack(fill="x")
        self.swatch_frame = ttk.Frame(f)
        self.swatch_frame.pack(fill="x")
        bf = ttk.Frame(f)
        bf.pack(fill="x", pady=(6, 0))
        self.btn_pick = ttk.Button(bf, text="이미지에서 범례 추출", command=self.toggle_legend_mode)
        self.btn_pick.pack(side="left", expand=True, fill="x")
        ttk.Button(bf, text="기본값", command=self.reset_colors).pack(side="left", padx=(4, 0))
        ttk.Label(f, text="색상칸 클릭=색 수정 / 추출=범례 막대를 드래그", foreground="#666").pack(anchor="w")

        # 판독 옵션
        f = ttk.LabelFrame(left, text="4. 판독 옵션", padding=6)
        f.pack(fill="x", pady=6)
        ttk.Label(f, text="색상 허용오차").grid(row=0, column=0, sticky="w")
        ttk.Scale(f, from_=5, to=120, variable=self.v_tol, command=lambda v: self.v_tol.set(int(float(v)))
                  ).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(f, textvariable=self.v_tol, width=4).grid(row=0, column=2)
        ttk.Label(f, text="샘플 크기(px)").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(f, values=[1, 3, 5, 7, 9, 15], width=5, state="readonly",
                     textvariable=self.v_sample).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Checkbutton(f, text="마우스 이동 시 실시간 값 표시", variable=self.v_hover).grid(
            row=2, column=0, columnspan=3, sticky="w")
        f.columnconfigure(1, weight=1)

        # 결과 패널
        f = ttk.LabelFrame(left, text="판독 결과", padding=6)
        f.pack(fill="x")
        self.res_swatch = tk.Label(f, text="", width=3, relief="solid", bd=1)
        self.res_swatch.grid(row=0, column=0, rowspan=2, padx=(0, 8), sticky="ns")
        self.res_value = tk.StringVar(value="-")
        self.res_info = tk.StringVar(value="이미지를 가져온 뒤 클릭하세요")
        tk.Label(f, textvariable=self.res_value, font=(FONT, 15, "bold"), anchor="w",
                 wraplength=250, justify="left").grid(row=0, column=1, sticky="w")
        tk.Label(f, textvariable=self.res_info, font=(FONT, 9), anchor="w", fg="#555",
                 justify="left", wraplength=250).grid(row=1, column=1, sticky="w")

        bf = ttk.Frame(left)
        bf.pack(fill="x", pady=6)
        ttk.Button(bf, text="마커 지우기", command=self.clear_markers).pack(side="left", expand=True, fill="x")
        ttk.Button(bf, text="결과 복사", command=self.copy_results).pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(bf, text="이미지 저장", command=self.save_image).pack(side="left", expand=True, fill="x")

        # 캔버스
        self.canvas = tk.Canvas(right, bg="#2b2b2b", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.StringVar(
            value="좌클릭: 값 고정 마커 | 휠: 확대/축소 | 드래그 또는 우클릭 드래그: 이동 | Backspace: 마지막 마커 삭제")
        ttk.Label(right, textvariable=self.status, padding=4).pack(fill="x")

        c = self.canvas
        c.bind("<Configure>", lambda e: self.render())
        c.bind("<Motion>", self._on_motion)
        c.bind("<Leave>", lambda e: c.delete("hover"))
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<ButtonPress-3>", self._on_pan_start)
        c.bind("<B3-Motion>", self._on_pan_move)
        c.bind("<MouseWheel>", self._on_wheel)

    def _bind_keys(self):
        self.bind("<F2>", lambda e: self.start_capture())
        self.bind("<Control-o>", lambda e: self.open_file())
        self.bind("<Control-v>", lambda e: self.paste())
        self.bind("<BackSpace>", lambda e: self._pop_marker())
        self.bind("<Escape>", lambda e: self._cancel_legend_mode())

    # ------------------------------------------------------------ legend
    def _build_legend(self, quiet: bool = True) -> Legend | None:
        try:
            return Legend(float(self.v_min.get()), float(self.v_max.get()), self.colors,
                          tolerance=self.v_tol.get(), open_ends=self.v_open.get(),
                          unit=self.v_unit.get().strip())
        except (ValueError, tk.TclError) as e:
            if not quiet:
                messagebox.showerror("입력 오류", str(e))
            return None

    def _on_legend_change(self):
        self._refresh_swatches()
        self.render()

    def _on_n_change(self):
        try:
            n = int(self.v_n.get())
        except (ValueError, tk.TclError):
            return
        if 2 <= n <= 30 and n != len(self.colors):
            self.colors = resample_colors(self.colors, n)
            self._on_legend_change()

    def _refresh_swatches(self):
        for w in self.swatch_frame.winfo_children():
            w.destroy()
        lg = self._build_legend()
        n = len(self.colors)
        for i in range(n - 1, -1, -1):
            row = tk.Frame(self.swatch_frame)
            row.pack(fill="x", pady=0)
            sw = tk.Label(row, bg=self.colors[i], width=6, height=1, relief="solid", bd=1, cursor="hand2")
            sw.pack(side="left")
            sw.bind("<Button-1>", lambda e, k=i: self._edit_color(k))
            txt = lg.short(i) if lg else "-"
            tk.Label(row, text=txt, font=(FONT, 9), width=18, anchor="w").pack(side="left", padx=6)
            tk.Label(row, text=self.colors[i], font=("Consolas", 8), fg="#888").pack(side="left")

    def _edit_color(self, i: int):
        _, hx = colorchooser.askcolor(color=self.colors[i], parent=self, title=f"구간 {i + 1} 색상")
        if hx:
            self.colors[i] = hx
            self._refresh_swatches()
            self.render()

    def reset_colors(self):
        self.colors = list(DEFAULT_COLORS)
        self.v_n.set(len(self.colors))
        self._on_legend_change()

    def toggle_legend_mode(self):
        if self.img is None:
            messagebox.showinfo("안내", "먼저 이미지를 가져오세요 (범례 막대가 포함된 이미지).")
            return
        self.legend_mode = not self.legend_mode
        self.btn_pick.config(text="추출 취소 (ESC)" if self.legend_mode else "이미지에서 범례 추출")
        self.status.set("범례 색상 막대를 드래그로 지정하세요 (세로: 위=최대, 가로: 오른쪽=최대)"
                        if self.legend_mode else "")

    def _cancel_legend_mode(self):
        if self.legend_mode:
            self.toggle_legend_mode()

    # ----------------------------------------------------------- image in
    def set_image(self, img: Image.Image):
        self.img = img.convert("RGB")
        self.arr = np.asarray(self.img)
        self.markers.clear()
        self.res_value.set("-")
        self.res_info.set("이미지를 클릭하면 값이 표시됩니다")
        self.res_swatch.config(bg=self.cget("bg"))
        self.fit()
        self.status.set(f"이미지 {self.img.width} x {self.img.height}px")

    def start_capture(self):
        try:
            delay = max(0, int(self.v_delay.get()))
        except (ValueError, tk.TclError):
            delay = 0
        self.withdraw()
        self.after(delay * 1000 + 250, self._do_capture)

    def _do_capture(self):
        try:
            img = capture.select_region(self)
        except Exception as e:  # noqa: BLE001
            self.deiconify()
            messagebox.showerror("캡처 오류", str(e))
            return
        self.deiconify()
        self.lift()
        self.focus_force()
        if img is not None:
            self.set_image(img)

    def open_file(self):
        p = filedialog.askopenfilename(
            parent=self, filetypes=[("이미지", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("모든 파일", "*.*")])
        if p:
            try:
                self.set_image(Image.open(p))
            except OSError as e:
                messagebox.showerror("열기 오류", str(e))

    def paste(self):
        data = ImageGrab.grabclipboard()
        if isinstance(data, Image.Image):
            self.set_image(data)
        elif isinstance(data, list) and data:
            try:
                self.set_image(Image.open(data[0]))
            except OSError:
                messagebox.showinfo("안내", "클립보드에 이미지가 없습니다.")
        else:
            messagebox.showinfo("안내", "클립보드에 이미지가 없습니다.")

    # ---------------------------------------------------------- viewport
    def fit(self):
        if self.img is None:
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.zoom = min(cw / self.img.width, ch / self.img.height, 1.0)
        self.ox = (self.img.width - cw / self.zoom) / 2
        self.oy = (self.img.height - ch / self.zoom) / 2
        self.render()

    def _to_img(self, cx, cy):
        return self.ox + cx / self.zoom, self.oy + cy / self.zoom

    def _to_canvas(self, ix, iy):
        return (ix - self.ox) * self.zoom, (iy - self.oy) * self.zoom

    def render(self):
        c = self.canvas
        c.delete("all")
        if self.img is None:
            c.create_text(c.winfo_width() // 2, c.winfo_height() // 2, fill="#aaa", font=(FONT, 14),
                          text="[화면 영역 캡처(F2)] 로 판독할 영역을 드래그하거나\n파일 열기 / 붙여넣기로 이미지를 가져오세요")
            return
        cw, ch, z = c.winfo_width(), c.winfo_height(), self.zoom
        x0, y0 = max(0.0, self.ox), max(0.0, self.oy)
        x1, y1 = min(float(self.img.width), self.ox + cw / z), min(float(self.img.height), self.oy + ch / z)
        if x1 > x0 and y1 > y0:
            ix0, iy0, ix1, iy1 = math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)
            crop = self.img.crop((ix0, iy0, ix1, iy1))
            size = (max(1, round((ix1 - ix0) * z)), max(1, round((iy1 - iy0) * z)))
            crop = crop.resize(size, Image.NEAREST if z >= 1 else Image.BILINEAR)
            self.photo = ImageTk.PhotoImage(crop)
            c.create_image((ix0 - self.ox) * z, (iy0 - self.oy) * z, anchor="nw", image=self.photo)

        lg = self._build_legend()
        for n, (mx, my) in enumerate(self.markers, 1):
            p = self._probe(mx, my, lg)
            sx, sy = self._to_canvas(mx + 0.5, my + 0.5)
            c.create_oval(sx - 7, sy - 7, sx + 7, sy + 7, outline="white", width=4)
            c.create_oval(sx - 7, sy - 7, sx + 7, sy + 7, outline="#e00000", width=2)
            c.create_line(sx - 12, sy, sx + 12, sy, fill="#e00000")
            c.create_line(sx, sy - 12, sx, sy + 12, fill="#e00000")
            self._label(sx, sy, [f"#{n}  ({mx}, {my})", p.text if p else "-"], p, "marker")

    def _label(self, sx, sy, lines, probe: Probe | None, tag):
        c = self.canvas
        border = "#e00000"
        if probe and probe.idx is not None:
            border = self.colors[probe.idx]
        text = "\n".join(lines)
        t = c.create_text(sx + 14, sy - 14, text=text, anchor="sw", font=(FONT, 11, "bold"),
                          fill="black", justify="left", tags=tag)
        x0, y0, x1, y1 = c.bbox(t)
        dx = dy = 0
        if x1 + 6 > c.winfo_width():
            dx = -(x1 - x0) - 28
        if y0 - 6 < 0:
            dy = (y1 - y0) + 28
        if dx or dy:
            c.move(t, dx, dy)
            x0, y0, x1, y1 = c.bbox(t)
        r = c.create_rectangle(x0 - 6, y0 - 4, x1 + 6, y1 + 4, fill="#fffde7", outline="white", width=5, tags=tag)
        r2 = c.create_rectangle(x0 - 6, y0 - 4, x1 + 6, y1 + 4, outline=border, width=3, tags=tag)
        c.tag_lower(r2, t)
        c.tag_lower(r, r2)

    # ------------------------------------------------------------- probe
    def _probe(self, ix: int, iy: int, lg: Legend | None) -> Probe | None:
        if self.arr is None or lg is None:
            return None
        h, w = self.arr.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return None
        r = int(self.v_sample.get()) // 2
        region = self.arr[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1]
        return lg.classify(region)

    def _show_result(self, ix, iy, p: Probe | None):
        if p is None:
            self.res_value.set("-")
            self.res_info.set("범례(최소/최대) 입력값을 확인하세요")
            return
        self.res_value.set(p.text)
        self.res_swatch.config(bg=self.colors[p.idx] if p.idx is not None else "#%02x%02x%02x" % p.rgb)
        extra = ""
        if p.idx is not None:
            extra = f"구간 {p.idx + 1}/{len(self.colors)} · 중앙 {fmt(p.center)} · 일치도 {p.confidence * 100:.0f}%\n"
        self.res_info.set(f"좌표 ({ix}, {iy}) · RGB {p.rgb}\n{extra}색상 거리 {p.distance:.1f}")

    # ------------------------------------------------------------ events
    def _on_motion(self, e):
        if self.img is None or self.legend_mode or not self.v_hover.get():
            return
        fx, fy = self._to_img(e.x, e.y)
        ix, iy = int(math.floor(fx)), int(math.floor(fy))
        self.canvas.delete("hover")
        if not (0 <= ix < self.img.width and 0 <= iy < self.img.height):
            return
        lg = self._build_legend()
        p = self._probe(ix, iy, lg)
        self._show_result(ix, iy, p)
        if p:
            short = lg.short(p.idx) if p.idx is not None else "범례 외"
            self._label(e.x, e.y, [short], p, "hover")

    def _on_press(self, e):
        self._press = (e.x, e.y, self.ox, self.oy, False)

    def _on_drag(self, e):
        if not self._press:
            return
        sx, sy, ox, oy, moved = self._press
        if not moved and math.hypot(e.x - sx, e.y - sy) < 4:
            return
        self._press = (sx, sy, ox, oy, True)
        if self.legend_mode:
            self.canvas.delete("legendbox")
            self.canvas.create_rectangle(sx, sy, e.x, e.y, outline="#00e5ff", width=2, dash=(4, 2), tags="legendbox")
        elif self.img is not None:
            self.ox = ox - (e.x - sx) / self.zoom
            self.oy = oy - (e.y - sy) / self.zoom
            self.render()

    def _on_release(self, e):
        if not self._press or self.img is None:
            self._press = None
            return
        sx, sy, _, _, moved = self._press
        self._press = None
        if self.legend_mode:
            if moved:
                self._extract_legend(sx, sy, e.x, e.y)
            return
        if moved:
            return
        fx, fy = self._to_img(e.x, e.y)
        ix, iy = int(math.floor(fx)), int(math.floor(fy))
        if 0 <= ix < self.img.width and 0 <= iy < self.img.height:
            self.markers.append((ix, iy))
            self._show_result(ix, iy, self._probe(ix, iy, self._build_legend()))
            self.render()

    def _extract_legend(self, cx0, cy0, cx1, cy1):
        a, b = self._to_img(cx0, cy0), self._to_img(cx1, cy1)
        try:
            cols = sample_colors(self.arr, (int(a[0]), int(a[1]), int(b[0]), int(b[1])), len(self.colors))
        except ValueError as ex:
            messagebox.showwarning("추출 실패", str(ex))
            return
        self.colors = cols
        self.legend_mode = False
        self.btn_pick.config(text="이미지에서 범례 추출")
        self.status.set("범례 색상을 추출했습니다.")
        self._on_legend_change()

    def _on_pan_start(self, e):
        self._pan = (e.x, e.y, self.ox, self.oy)

    def _on_pan_move(self, e):
        if self.img is None:
            return
        sx, sy, ox, oy = self._pan
        self.ox = ox - (e.x - sx) / self.zoom
        self.oy = oy - (e.y - sy) / self.zoom
        self.render()

    def _on_wheel(self, e):
        if self.img is None:
            return
        fx, fy = self._to_img(e.x, e.y)
        z = self.zoom * (1.25 if e.delta > 0 else 0.8)
        self.zoom = min(32.0, max(0.05, z))
        self.ox = fx - e.x / self.zoom
        self.oy = fy - e.y / self.zoom
        self.render()

    # ----------------------------------------------------------- markers
    def clear_markers(self):
        self.markers.clear()
        self.render()

    def _pop_marker(self):
        if self.markers and isinstance(self.focus_get(), (tk.Canvas, tk.Tk)):
            self.markers.pop()
            self.render()

    def _marker_lines(self):
        lg = self._build_legend()
        out = []
        for n, (x, y) in enumerate(self.markers, 1):
            p = self._probe(x, y, lg)
            out.append((n, x, y, p.text if p else "-"))
        return out

    def copy_results(self):
        rows = self._marker_lines()
        if not rows:
            return
        self.clipboard_clear()
        self.clipboard_append("\n".join(f"#{n}\t({x},{y})\t{t}" for n, x, y, t in rows))
        self.status.set(f"마커 {len(rows)}개 결과를 클립보드에 복사했습니다.")

    def save_image(self):
        if self.img is None:
            return
        p = filedialog.asksaveasfilename(parent=self, defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not p:
            return
        im = self.img.copy()
        d = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("malgun.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        for n, x, y, t in self._marker_lines():
            d.ellipse((x - 6, y - 6, x + 6, y + 6), outline="red", width=2)
            label = f"#{n} {t}"
            bb = d.textbbox((x + 10, y - 22), label, font=font)
            d.rectangle((bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2), fill="#fffde7", outline="red")
            d.text((x + 10, y - 22), label, fill="black", font=font)
        im.save(p)
        self.status.set(f"저장됨: {p}")

    # ------------------------------------------------------------- close
    def _on_close(self):
        try:
            settings.save({
                "vmin": self.v_min.get(), "vmax": self.v_max.get(), "unit": self.v_unit.get(),
                "colors": self.colors, "tolerance": self.v_tol.get(), "sample": self.v_sample.get(),
                "open_ends": self.v_open.get(), "delay": self.v_delay.get(), "hover": self.v_hover.get(),
            })
        except (OSError, tk.TclError):
            pass
        self.destroy()


def main():
    App().mainloop()
