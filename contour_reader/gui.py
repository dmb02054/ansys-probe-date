"""컨투어 이미지 값 판독기 GUI."""
from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import colorchooser, filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageGrab, ImageTk

from . import capture, settings
from .export import ExportMarker, auto_offset, compose
from .interpolate import refine_probe
from .legend import (DEFAULT_COLORS, Legend, Probe, fmt, hex_to_rgb,
                     resample_colors, sample_colors)

FONT = "Malgun Gothic"
ACCENT = "#3b82f6"
GRAY_BTN = "#6b7280"


@dataclass
class Marker:
    x: int
    y: int
    dx: float | None = None   # 라벨 좌상단의 마커 기준 오프셋(이미지 px). None=자동 배치
    dy: float | None = None


def _text_color(hex_color: str) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return "#000000" if (r * 299 + g * 587 + b * 114) / 1000 > 140 else "#ffffff"


class App(ctk.CTk):
    def __init__(self):
        capture.enable_dpi_awareness()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        super().__init__()
        self.title("컨투어 값 판독기")
        self.geometry("980x640")
        self.minsize(820, 560)

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
        self.v_interp = tk.BooleanVar(value=bool(cfg["interp"]))
        self.v_search = tk.IntVar(value=int(cfg["search"]))
        self.v_black = tk.BooleanVar(value=bool(cfg["ignore_black"]))
        self.v_black_thr = tk.IntVar(value=int(cfg["black_thr"]))
        self.v_shade = tk.IntVar(value=int(cfg["shade"]))
        self.edge_vars: list[tk.StringVar] = []
        self._saved_edges: dict | None = cfg.get("edges") or {}
        self.img: Image.Image | None = None
        self.arr: np.ndarray | None = None
        self.photo = None
        self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0
        self.markers: list[Marker] = []
        self.selected: int | None = None
        self.legend_mode = False
        self._press = None
        self._drag_label = None
        self._drag_rect = None
        self._bmap_cache: np.ndarray | None = None
        self._probe_cache: dict[tuple[int, int], Probe] = {}

        self._label_boxes: dict[int, tuple] = {}
        self._build_ui()
        self._bind_keys()
        for v in (self.v_min, self.v_max, self.v_unit, self.v_tol, self.v_sample, self.v_open,
                  self.v_interp, self.v_search, self.v_black, self.v_black_thr, self.v_shade):
            v.trace_add("write", lambda *_: self._on_legend_change())
        self._refresh_swatches()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- UI
    def _btn(self, parent, text, cmd, **kw):
        return ctk.CTkButton(parent, text=text, command=cmd, height=26, font=self.f_s, **kw)

    def _check(self, parent, text, var):
        return ctk.CTkCheckBox(parent, text=text, variable=var, font=self.f_s, checkbox_width=18,
                               checkbox_height=18, border_width=2)

    def _field(self, parent, text, var, width=44):
        ctk.CTkLabel(parent, text=text, font=self.f_s).pack(side="left")
        ctk.CTkEntry(parent, textvariable=var, width=width, height=24, font=self.f_s,
                     justify="center").pack(side="left", padx=(4, 10))

    def _slider_row(self, parent, text, var, lo, hi):
        r = ctk.CTkFrame(parent, fg_color="transparent")
        r.pack(fill="x", pady=1)
        ctk.CTkLabel(r, text=text, font=self.f_s, width=84, anchor="w").pack(side="left")
        val = ctk.CTkLabel(r, text=str(var.get()), font=self.f_s, width=28)
        val.pack(side="right")

        def on(v):
            var.set(int(round(v)))
            val.configure(text=str(int(round(v))))
        s = ctk.CTkSlider(r, from_=lo, to=hi, number_of_steps=hi - lo, height=14, command=on)
        s.set(var.get())
        s.pack(side="left", fill="x", expand=True, padx=4)

    def _build_ui(self):
        self.f_s = ctk.CTkFont(family=FONT, size=12)
        self.f_b = ctk.CTkFont(family=FONT, size=13, weight="bold")
        left = ctk.CTkFrame(self, width=270, corner_radius=0)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)   # 내용 길이가 바뀌어도 좌측 패널 크기가 흔들리지 않게 고정
        right = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)

        # 아래쪽 고정 영역 (먼저 pack 해야 창이 작아져도 잘리지 않음)
        bf = ctk.CTkFrame(left, fg_color="transparent")
        bf.pack(side="bottom", fill="x", padx=8, pady=(4, 8))
        for text, cmd in (("마커 지우기", self.clear_markers), ("결과 복사", self.copy_results),
                          ("이미지 저장", self.save_image)):
            self._btn(bf, text, cmd, fg_color=GRAY_BTN, hover_color="#4b5563", width=60).pack(
                side="left", expand=True, fill="x", padx=2)
        self._opt_open = False
        self.btn_opt = ctk.CTkButton(left, text="⚙ 판독 옵션  ▾", command=self._toggle_options, height=26,
                                     font=self.f_s, fg_color="transparent", text_color="#374151",
                                     hover_color="#e5e7eb", border_width=1, border_color="#9ca3af")
        self.btn_opt.pack(side="bottom", fill="x", padx=10, pady=(4, 0))

        # 판독 옵션 (더보기로 펼침)
        f = self.opt_frame = ctk.CTkFrame(left, fg_color="#f3f4f6")
        self._slider_row(f, "색상 허용오차", self.v_tol, 5, 120)
        self._slider_row(f, "그림자/조명 보정", self.v_shade, 0, 100)
        r = ctk.CTkFrame(f, fg_color="transparent")
        r.pack(fill="x", pady=2)
        ctk.CTkLabel(r, text="샘플(px)", font=self.f_s).pack(side="left")
        self._sample_menu = ctk.CTkOptionMenu(r, values=["1", "3", "5", "7", "9", "15"], width=56, height=24,
                                              font=self.f_s, dropdown_font=self.f_s,
                                              command=lambda s: self.v_sample.set(int(s)))
        self._sample_menu.set(str(self.v_sample.get()))
        self._sample_menu.pack(side="left", padx=(4, 10))
        self._field(r, "탐색반경", self.v_search)
        r = ctk.CTkFrame(f, fg_color="transparent")
        r.pack(fill="x", pady=2)
        self._field(r, "캡처 지연(초)", self.v_delay)
        r = ctk.CTkFrame(f, fg_color="transparent")
        r.pack(fill="x", pady=2)
        self._check(r, "검은선 무시", self.v_black).pack(side="left")
        self._field(r, "임계", self.v_black_thr)
        r = ctk.CTkFrame(f, fg_color="transparent")
        r.pack(fill="x", pady=2)
        self._check(r, "소수 보간", self.v_interp).pack(side="left", padx=(0, 10))
        self._check(r, "실시간 값", self.v_hover).pack(side="left")
        r = ctk.CTkFrame(f, fg_color="transparent")
        r.pack(fill="x", pady=(2, 6))
        self._check(r, "양 끝 구간을 이상/이하로 표시", self.v_open).pack(side="left")

        # 이미지 입력
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(8, 4))
        for text, cmd in (("캡처 F2", self.start_capture), ("열기", self.open_file), ("붙여넣기", self.paste)):
            self._btn(row, text, cmd, width=60).pack(side="left", expand=True, fill="x", padx=2)

        # 단위 / 구간 수
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        self._field(row, "단위", self.v_unit, width=52)
        ctk.CTkLabel(row, text="구간 수", font=self.f_s).pack(side="left")
        self.n_menu = ctk.CTkOptionMenu(row, values=[str(i) for i in range(2, 31)], width=56, height=24,
                                        font=self.f_s, dropdown_font=self.f_s, command=self._on_n_menu)
        self.n_menu.set(str(self.v_n.get()))
        self.n_menu.pack(side="left", padx=4)

        # 범례 값 / 색상: 칸에 경계값을 입력(비우면 자동 보간), 색상칸 클릭=색 수정
        ctk.CTkLabel(left, text="범례 값 · 색상 (위=최대)", font=self.f_s, text_color="#6b7280",
                     anchor="w").pack(fill="x", padx=12, pady=(6, 0))
        bf = ctk.CTkFrame(left, fg_color="transparent")
        bf.pack(side="bottom", fill="x", padx=8, pady=(4, 0))
        self.btn_pick = self._btn(bf, "범례 추출", self.toggle_legend_mode, width=60)
        self.btn_pick.pack(side="left", expand=True, fill="x", padx=2)
        self._btn(bf, "기본 색", self.reset_colors, width=60, fg_color=GRAY_BTN,
                  hover_color="#4b5563").pack(side="left", expand=True, fill="x", padx=2)
        self._btn(bf, "값 비우기", self.clear_edges, width=60, fg_color=GRAY_BTN,
                  hover_color="#4b5563").pack(side="left", expand=True, fill="x", padx=2)
        self.swatch_frame = ctk.CTkScrollableFrame(left, height=230, fg_color="transparent")
        self.swatch_frame.pack(fill="both", expand=True, padx=4, pady=2)
        self._swatch_rows: list[tuple] = []

        self.canvas = tk.Canvas(right, bg="#2b2b2b", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        # 판독 결과: 캔버스 우측 상단에 작게 (색+값), ▾ 로 자세한 내용 펼침
        self.res_box = ctk.CTkFrame(self.canvas, fg_color="white", corner_radius=8, border_width=1,
                                    border_color="#d1d5db")
        self.res_box.place(relx=1.0, x=-8, y=8, anchor="ne")
        top = ctk.CTkFrame(self.res_box, fg_color="transparent")
        top.pack(padx=6, pady=4)
        self.res_swatch = ctk.CTkLabel(top, text="", width=22, height=22, corner_radius=4, fg_color="#cccccc")
        self.res_swatch.pack(side="left")
        self.res_value = tk.StringVar(value="-")
        self.res_info = tk.StringVar(value="이미지를 클릭하세요")
        ctk.CTkLabel(top, textvariable=self.res_value, font=self.f_b, text_color="#111827").pack(
            side="left", padx=6)
        self.btn_res = ctk.CTkButton(top, text="▾", width=22, height=22, fg_color="transparent",
                                     text_color="#555555", hover_color="#e5e7eb", command=self._toggle_result)
        self.btn_res.pack(side="left")
        self.res_info_lbl = ctk.CTkLabel(self.res_box, textvariable=self.res_info, font=self.f_s,
                                         text_color="#4b5563", justify="left", anchor="w", wraplength=300)
        self._res_open = False

        self.status = tk.StringVar(
            value="좌클릭: 마커 추가/선택 | 라벨 드래그: 위치 이동 | 휠: 확대/축소 | 우클릭 드래그: 이동 | Backspace: 선택 마커 삭제")
        ctk.CTkLabel(right, textvariable=self.status, font=ctk.CTkFont(family=FONT, size=11),
                     text_color="#4b5563", anchor="w").pack(fill="x", padx=6, pady=3)

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

    def _toggle_options(self):
        self._opt_open = not self._opt_open
        if self._opt_open:
            self.opt_frame.pack(side="bottom", fill="x", padx=8, pady=(2, 4), before=self.btn_opt)
        else:
            self.opt_frame.pack_forget()
        self.btn_opt.configure(text="⚙ 판독 옵션  ▴" if self._opt_open else "⚙ 판독 옵션  ▾")

    def _toggle_result(self):
        self._res_open = not self._res_open
        if self._res_open:
            self.res_info_lbl.pack(fill="x", padx=8, pady=(0, 6))
        else:
            self.res_info_lbl.pack_forget()
        self.btn_res.configure(text="▴" if self._res_open else "▾")

    def _on_n_menu(self, s: str):
        self.v_n.set(int(s))
        self._on_n_change()

    def _bind_keys(self):
        self.bind("<F2>", lambda e: self.start_capture())
        self.bind("<Control-o>", lambda e: self.open_file())
        self.bind("<Control-v>", lambda e: self.paste())
        self.bind("<BackSpace>", lambda e: self._pop_marker())
        self.bind("<Escape>", lambda e: self._cancel_legend_mode())

    # ------------------------------------------------------------ legend
    def _build_legend(self, quiet: bool = True) -> Legend | None:
        try:
            n = len(self.colors)
            edges = {}
            if len(self.edge_vars) == n + 1:
                for k in range(1, n):
                    s = self.edge_vars[k].get().strip()
                    if s:
                        edges[k] = float(s)
            return Legend(float(self.v_min.get()), float(self.v_max.get()), self.colors,
                          tolerance=self.v_tol.get(), open_ends=self.v_open.get(),
                          unit=self.v_unit.get().strip(), ignore_black=self.v_black.get(),
                          black_threshold=self.v_black_thr.get(), edges=edges,
                          shade=self.v_shade.get())
        except (ValueError, tk.TclError) as e:
            if not quiet:
                messagebox.showerror("입력 오류", str(e))
            return None

    def _on_legend_change(self):
        self._bmap_cache = None
        self._probe_cache.clear()
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

    def _make_edge_vars(self, n: int, saved: dict | None = None):
        """경계값 입력 변수: [0]=최소값, [n]=최대값(상단 입력칸과 공유), 나머지는 비워두면 자동 보간."""
        saved = saved or {}
        self.edge_vars = [self.v_min]
        for k in range(1, n):
            var = tk.StringVar(value=str(saved.get(str(k), saved.get(k, ""))))
            var.trace_add("write", lambda *_: self._on_legend_change())
            self.edge_vars.append(var)
        self.edge_vars.append(self.v_max)

    def clear_edges(self):
        for var in self.edge_vars[1:-1]:
            var.set("")

    def _refresh_swatches(self):
        n = len(self.colors)
        rebuild = len(self._swatch_rows) != n
        if rebuild:   # 구간 수가 바뀔 때만 위젯을 다시 만든다 (깜빡임/흔들림 방지)
            self._make_edge_vars(n, self._saved_edges)
            self._saved_edges = None
            for w in self.swatch_frame.winfo_children():
                w.destroy()
            self._swatch_rows = []
            for i in range(n - 1, -1, -1):
                row = ctk.CTkFrame(self.swatch_frame, fg_color="transparent")
                row.pack(fill="x", pady=1)
                ctk.CTkEntry(row, textvariable=self.edge_vars[i + 1], width=64, height=22, font=self.f_s,
                             justify="right").pack(side="left")
                sw = ctk.CTkButton(row, text="", width=34, height=18, corner_radius=3, border_width=1,
                                   border_color="#6b7280", hover=False,
                                   command=lambda k=i: self._edit_color(k))
                sw.pack(side="left", padx=(6, 0))
                t1 = ctk.CTkLabel(row, text="", font=self.f_s, anchor="w", text_color="#4b5563")
                t1.pack(side="left", padx=6)
                self._swatch_rows.append((i, sw, t1))
            row = ctk.CTkFrame(self.swatch_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkEntry(row, textvariable=self.edge_vars[0], width=64, height=22, font=self.f_s,
                         justify="right").pack(side="left")
            ctk.CTkLabel(row, text="← 최소", font=self.f_s, text_color="#9ca3af").pack(side="left", padx=6)
        lg = self._build_legend()
        for i, sw, t1 in self._swatch_rows:
            sw.configure(fg_color=self.colors[i])
            t1.configure(text=lg.short(i) if lg else "-")

    def _edit_color(self, i: int):
        _, hx = colorchooser.askcolor(color=self.colors[i], parent=self, title=f"구간 {i + 1} 색상")
        if hx:
            self.colors[i] = hx
            self._on_legend_change()

    def reset_colors(self):
        self.colors = list(DEFAULT_COLORS)
        self.v_n.set(len(self.colors))
        self.n_menu.set(str(len(self.colors)))
        self._on_legend_change()

    def toggle_legend_mode(self):
        if self.img is None:
            messagebox.showinfo("안내", "먼저 이미지를 가져오세요 (범례 막대가 포함된 이미지).")
            return
        self.legend_mode = not self.legend_mode
        self.btn_pick.configure(text="추출 취소(ESC)" if self.legend_mode else "범례 추출")
        self.status.set("범례 색상 막대를 드래그로 지정하세요 (세로: 위=최대, 가로: 오른쪽=최대)"
                        if self.legend_mode else "")

    def _cancel_legend_mode(self):
        if self.legend_mode:
            self.toggle_legend_mode()

    # ----------------------------------------------------------- image in
    def set_image(self, img: Image.Image):
        self.img = img.convert("RGB")
        self.arr = np.asarray(self.img)
        self._bmap_cache = None
        self._probe_cache.clear()
        self.markers.clear()
        self.selected = None
        self.res_value.set("-")
        self.res_info.set("이미지를 클릭하세요")
        self.res_swatch.configure(fg_color="#cccccc")
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
            c.create_image((ix0 - self.ox) * z, (iy0 - self.oy) * z, anchor="nw", image=self.photo, tags="img")
        self.draw_overlay()

    def draw_overlay(self):
        """마커/라벨/연결선만 다시 그린다 (라벨 드래그 시 빠른 갱신)."""
        c = self.canvas
        c.delete("ov", "hover")
        if self.img is None:
            return
        lg = self._build_legend()
        unit = f" {lg.unit}" if lg and lg.unit else ""
        cw, ch = max(1, c.winfo_width()), max(1, c.winfo_height())
        self._label_boxes = {}
        for i, m in enumerate(self.markers):
            p = self._probe(m.x, m.y, lg)
            sx, sy = self._to_canvas(m.x + 0.5, m.y + 0.5)
            sel = i == self.selected
            border = self.colors[p.idx] if p and p.idx is not None else "#e00000"
            short = p.label if (p and lg) else "-"
            u = unit if p and p.idx is not None else ""
            tag = f"lbl{i}"
            t = c.create_text(0, 0, text=f"#{i + 1}  {short}{u}", anchor="nw", font=(FONT, 11, "bold"))
            x0, y0, x1, y1 = c.bbox(t)
            c.delete(t)
            chip, pad = 14, 6
            bw, bh = pad * 2 + chip + 6 + (x1 - x0), max(chip, y1 - y0) + pad * 2
            if m.dx is None:
                dx, dy = auto_offset(bw, bh, cw, ch, sx, sy, 16)
            else:
                dx, dy = m.dx * self.zoom, m.dy * self.zoom
            bx, by = sx + dx, sy + dy
            self._label_boxes[i] = (bx, by, bw, bh)
            nx, ny = min(max(sx, bx), bx + bw), min(max(sy, by), by + bh)
            if (nx, ny) != (sx, sy):
                c.create_line(sx, sy, nx, ny, fill="white", width=4, tags="ov")
                c.create_line(sx, sy, nx, ny, fill="#333333", width=2, tags="ov")
            c.create_oval(sx - 7, sy - 7, sx + 7, sy + 7, outline="white", width=4, tags="ov")
            c.create_oval(sx - 7, sy - 7, sx + 7, sy + 7, outline="#ff1744" if sel else "#e00000", width=3 if sel else 2,
                          tags="ov")
            c.create_line(sx - 12, sy, sx + 12, sy, fill="#e00000", tags="ov")
            c.create_line(sx, sy - 12, sx, sy + 12, fill="#e00000", tags="ov")
            c.create_rectangle(bx - 2, by - 2, bx + bw + 2, by + bh + 2, fill="white", outline="white", tags=("ov", tag))
            c.create_rectangle(bx, by, bx + bw, by + bh, fill="white", outline=border, width=4 if sel else 3,
                               tags=("ov", tag))
            cy = by + bh / 2
            c.create_rectangle(bx + pad, cy - chip / 2, bx + pad + chip, cy + chip / 2, fill=border if p and p.idx is not None
                               else "#cccccc", outline="#555555", tags=("ov", tag))
            c.create_text(bx + pad + chip + 6, cy, text=f"#{i + 1}  {short}{u}", anchor="w",
                          font=(FONT, 11, "bold"), fill="#111111", tags=("ov", tag))

    def _tooltip(self, sx, sy, text, probe: Probe | None):
        c = self.canvas
        border = self.colors[probe.idx] if probe and probe.idx is not None else "#e00000"
        t = c.create_text(sx + 14, sy - 14, text=text, anchor="sw", font=(FONT, 11, "bold"),
                          fill="black", tags="hover")
        x0, y0, x1, y1 = c.bbox(t)
        dx = dy = 0
        if x1 + 6 > c.winfo_width():
            dx = -(x1 - x0) - 28
        if y0 - 6 < 0:
            dy = (y1 - y0) + 28
        if dx or dy:
            c.move(t, dx, dy)
            x0, y0, x1, y1 = c.bbox(t)
        r = c.create_rectangle(x0 - 6, y0 - 4, x1 + 6, y1 + 4, fill="#fffde7", outline="white", width=5, tags="hover")
        r2 = c.create_rectangle(x0 - 6, y0 - 4, x1 + 6, y1 + 4, outline=border, width=3, tags="hover")
        c.tag_lower(r2, t)
        c.tag_lower(r, r2)

    # ------------------------------------------------------------- probe
    def _probe(self, ix: int, iy: int, lg: Legend | None) -> Probe | None:
        if self.arr is None or lg is None:
            return None
        h, w = self.arr.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return None
        key = (ix, iy)
        if key in self._probe_cache:
            return self._probe_cache[key]
        r0 = int(self.v_sample.get()) // 2
        # 검은 선 위를 눌러도 인접한 색으로 판단하도록, 검은색뿐이면 샘플 영역을 조금씩 넓힌다
        for r in range(r0, r0 + 7):
            region = self.arr[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1]
            if not lg.black_mask(region).all():
                break
        p = lg.classify(region)
        if self.v_interp.get() and p.idx is not None:
            try:
                radius = min(200, max(10, int(self.v_search.get())))
            except (ValueError, tk.TclError):
                radius = 60
            refine_probe(p, self.arr, ix, iy, lg, radius, self._band_map(lg))
        if len(self._probe_cache) > 5000:
            self._probe_cache.clear()
        self._probe_cache[key] = p
        return p

    def _band_map(self, lg: Legend) -> np.ndarray | None:
        """이미지 전체 구간 번호 캐시 (큰 이미지는 창 단위로 계산하도록 None)."""
        if self.arr.shape[0] * self.arr.shape[1] > 4_000_000:
            return None
        if self._bmap_cache is None:
            self._bmap_cache = lg.band_map(self.arr)
        return self._bmap_cache

    def _show_result(self, ix, iy, p: Probe | None):
        if p is None:
            self.res_value.set("-")
            self.res_info.set("범례 입력값을 확인하세요")
            return
        self.res_value.set(p.text)
        self.res_swatch.configure(fg_color=self.colors[p.idx] if p.idx is not None else "#%02x%02x%02x" % p.rgb)
        lines = [f"좌표 ({ix}, {iy}) · RGB {p.rgb}"]
        if p.idx is not None:
            lo, hi = self.legend_range(p.idx)
            lines.append(f"구간 {p.idx + 1}/{len(self.colors)} ({lo} ~ {hi}) · 일치도 {p.confidence * 100:.0f}%")
            if p.kind == "interp":
                lines.append(f"아래 경계 {p.d_lo:.1f}px · 위 경계 {p.d_hi:.1f}px 기준 보간")
            elif p.kind == "center":
                lines.append("경계를 못 찾아 구간 중앙값으로 근사")
            elif p.kind == "open":
                lines.append("최대/최소 밖 값도 같은 색이라 보간 불가")
        if math.isfinite(p.distance):
            lines.append(f"색상 거리 {p.distance:.1f}")
        self.res_info.set("\n".join(lines))

    def legend_range(self, i: int) -> tuple[str, str]:
        lg = self._build_legend()
        lo, hi = lg.band_range(i) if lg else (0, 0)
        return fmt(lo), fmt(hi)

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
            u = f" {lg.unit}" if lg.unit and p.idx is not None else ""
            self._tooltip(e.x, e.y, p.label + u, p)

    def _on_press(self, e):
        self.canvas.focus_set()   # 입력칸에 포커스가 있어도 Backspace로 마커를 지울 수 있게
        self._press = (e.x, e.y, self.ox, self.oy, False)
        self._drag_label = None
        if self.img is None or self.legend_mode:
            return
        for it in reversed(self.canvas.find_overlapping(e.x, e.y, e.x, e.y)):
            tag = next((t for t in self.canvas.gettags(it) if t.startswith("lbl")), None)
            if tag:
                i = int(tag[3:])
                bx, by, _, _ = self._label_boxes[i]
                sx, sy = self._to_canvas(self.markers[i].x + 0.5, self.markers[i].y + 0.5)
                self._drag_label = (i, e.x, e.y, bx - sx, by - sy)
                if self.selected != i:
                    self.select(i)
                return

    def _on_drag(self, e):
        if not self._press:
            return
        sx, sy, ox, oy, moved = self._press
        if not moved and math.hypot(e.x - sx, e.y - sy) < 4:
            return
        self._press = (sx, sy, ox, oy, True)
        if self._drag_label is not None:
            i, px, py, off_x, off_y = self._drag_label
            self.markers[i].dx = (off_x + e.x - px) / self.zoom
            self.markers[i].dy = (off_y + e.y - py) / self.zoom
            self.draw_overlay()
        elif self.legend_mode:
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
        if self._drag_label is not None:
            self._drag_label = None
            return
        if self.legend_mode:
            if moved:
                self._extract_legend(sx, sy, e.x, e.y)
            return
        if moved:
            return
        hit = self._hit_marker(e.x, e.y)
        if hit is not None:
            self.select(hit)
            return
        fx, fy = self._to_img(e.x, e.y)
        ix, iy = int(math.floor(fx)), int(math.floor(fy))
        if 0 <= ix < self.img.width and 0 <= iy < self.img.height:
            self.markers.append(Marker(ix, iy))
            self.select(len(self.markers) - 1)

    def _hit_marker(self, cx, cy) -> int | None:
        for i in range(len(self.markers) - 1, -1, -1):
            sx, sy = self._to_canvas(self.markers[i].x + 0.5, self.markers[i].y + 0.5)
            if math.hypot(cx - sx, cy - sy) <= 10:
                return i
        return None

    def select(self, i: int | None):
        self.selected = i
        if i is not None:
            m = self.markers[i]
            self._show_result(m.x, m.y, self._probe(m.x, m.y, self._build_legend()))
        self.draw_overlay()

    def _extract_legend(self, cx0, cy0, cx1, cy1):
        a, b = self._to_img(cx0, cy0), self._to_img(cx1, cy1)
        try:
            cols = sample_colors(self.arr, (int(a[0]), int(a[1]), int(b[0]), int(b[1])), len(self.colors))
        except ValueError as ex:
            messagebox.showwarning("추출 실패", str(ex))
            return
        self.colors = cols
        self.legend_mode = False
        self.n_menu.set(str(len(self.colors)))
        self.btn_pick.configure(text="범례 추출")
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
        self.select(None)

    def _pop_marker(self):
        if self.markers and isinstance(self.focus_get(), (tk.Canvas, tk.Tk)):
            self.markers.pop(self.selected if self.selected is not None else -1)
            self.select(len(self.markers) - 1 if self.markers else None)

    def _marker_lines(self):
        lg = self._build_legend()
        out = []
        for n, m in enumerate(self.markers, 1):
            p = self._probe(m.x, m.y, lg)
            out.append((n, m.x, m.y, p.text if p else "-"))
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
        lg = self._build_legend(quiet=False)
        if lg is None:
            return
        p = filedialog.asksaveasfilename(parent=self, defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not p:
            return
        items = []
        for i, m in enumerate(self.markers):
            pr = self._probe(m.x, m.y, lg)
            items.append(ExportMarker(i + 1, m.x, m.y, pr.label if pr else "-",
                                      self.colors[pr.idx] if pr and pr.idx is not None else None,
                                      None if m.dx is None else (m.dx, m.dy)))
        compose(self.img, items, lg.unit, None if self.selected is None else self.selected + 1).save(p)
        self.status.set(f"저장됨: {p}")

    # ------------------------------------------------------------- close
    def _on_close(self):
        try:
            settings.save({
                "vmin": self.v_min.get(), "vmax": self.v_max.get(), "unit": self.v_unit.get(),
                "colors": self.colors, "tolerance": self.v_tol.get(), "sample": self.v_sample.get(),
                "open_ends": self.v_open.get(), "delay": self.v_delay.get(), "hover": self.v_hover.get(),
                "interp": self.v_interp.get(), "search": self.v_search.get(),
                "ignore_black": self.v_black.get(), "black_thr": self.v_black_thr.get(),
                "shade": self.v_shade.get(),
                "edges": {str(k): v.get().strip() for k, v in enumerate(self.edge_vars)
                          if 0 < k < len(self.edge_vars) - 1 and v.get().strip()},
            })
        except (OSError, tk.TclError):
            pass
        self.destroy()


def main():
    App().mainloop()
