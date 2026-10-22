"""컨투어 이미지 값 판독기 GUI (CustomTkinter).

화면 구성
  - 왼쪽 패널 : 이미지 가져오기, 단위/구간 수, 범례 값·색상 목록, 판독 옵션(접힘), 마커 지우기/복사/저장
  - 오른쪽    : 이미지 캔버스 (우측 상단에 판독 결과 상자, 아래에 상태 표시줄)
이미지 표시/마커는 일반 tk.Canvas 로 직접 그리고, 나머지 위젯은 CustomTkinter 를 사용한다.
"""
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
from .legend import (DEFAULT_COLORS, Legend, Probe, fmt,
                     resample_colors, sample_colors)

FONT = "Malgun Gothic"
GRAY_BTN = "#6b7280"    # 보조 버튼 색


@dataclass
class Marker:
    """이미지 위에 찍은 판독 지점."""
    x: int                       # 이미지 좌표(px)
    y: int
    dx: float | None = None      # 라벨 좌상단의 마커 기준 오프셋(이미지 px). None=자동 배치
    dy: float | None = None


class App(ctk.CTk):
    """메인 창. 입력값(Tk 변수) -> Legend 생성 -> 클릭 지점 판독 -> 화면/마커 갱신 흐름으로 동작한다."""

    def __init__(self):
        capture.enable_dpi_awareness()   # 창 생성 전에 호출해야 화면 캡처 좌표가 정확하다
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        super().__init__()
        self.title("컨투어 값 판독기")
        self.geometry("980x640")
        self.minsize(820, 560)

        # 저장된 설정을 읽어 입력 변수(Tk 변수)에 채운다
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
        # 중간 경계값 입력 변수 목록(구간 수 + 1개). 위젯을 만들 때 _make_edge_vars 에서 채운다
        self.edge_vars: list[tk.StringVar] = []
        self._saved_edges: dict | None = cfg.get("edges") or {}

        # 이미지/화면 상태
        self.img: Image.Image | None = None      # 원본 이미지
        self.arr: np.ndarray | None = None       # 원본 이미지의 numpy 배열 (판독용)
        self.photo = None                        # 캔버스에 그린 PhotoImage (참조를 유지해야 사라지지 않음)
        self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0   # 확대 배율, 화면 좌상단에 해당하는 이미지 좌표
        self.markers: list[Marker] = []
        self.selected: int | None = None         # 선택된 마커 번호
        self.legend_mode = False                 # True 이면 드래그로 범례 막대 색상을 추출하는 중
        self._press = None                       # 마우스 누른 상태 (x, y, ox, oy, 이동 여부)
        self._pan = None                         # 우클릭 이동(패닝) 시작 상태
        self._drag_label = None                  # 라벨을 드래그하는 중이면 그 정보
        self._bmap_cache: np.ndarray | None = None   # 이미지 전체 구간 번호 캐시 (범례가 바뀌면 무효화)
        self._legend_cache: Legend | None = None     # 현재 입력으로 만든 Legend (범례가 바뀌면 무효화)
        self._probe_cache: dict[tuple[int, int], Probe] = {}   # 지점별 판독 결과 캐시

        self._label_boxes: dict[int, tuple] = {}   # 마커 라벨 박스 위치 (클릭/드래그 판정용)
        self._build_ui()
        self._bind_keys()
        # 판독에 영향을 주는 입력이 바뀌면 캐시를 비우고 화면을 다시 그린다
        for v in (self.v_min, self.v_max, self.v_unit, self.v_tol, self.v_sample, self.v_open,
                  self.v_interp, self.v_search, self.v_black, self.v_black_thr, self.v_shade):
            v.trace_add("write", lambda *_: self._on_legend_change())
        self._refresh_swatches()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- UI
    def _btn(self, parent, text, cmd, **kw):
        """왼쪽 패널용 작은 버튼."""
        return ctk.CTkButton(parent, text=text, command=cmd, height=26, font=self.f_s, **kw)

    def _check(self, parent, text, var):
        """작은 체크박스."""
        return ctk.CTkCheckBox(parent, text=text, variable=var, font=self.f_s, checkbox_width=18,
                               checkbox_height=18, border_width=2)

    def _field(self, parent, text, var, width=44):
        """'라벨 + 짧은 입력칸' 한 쌍을 한 줄에 나란히 배치한다."""
        ctk.CTkLabel(parent, text=text, font=self.f_s).pack(side="left")
        ctk.CTkEntry(parent, textvariable=var, width=width, height=24, font=self.f_s,
                     justify="center").pack(side="left", padx=(4, 10))

    def _slider_row(self, parent, text, var, lo, hi):
        """정수 슬라이더 한 줄 (라벨 + 슬라이더 + 현재 값)."""
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
        """전체 화면 구성. 위젯 생성 순서가 pack 배치 순서에 영향을 주므로 순서를 바꾸지 말 것."""
        self.f_s = ctk.CTkFont(family=FONT, size=12)
        self.f_b = ctk.CTkFont(family=FONT, size=13, weight="bold")
        left = ctk.CTkFrame(self, width=270, corner_radius=0)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)   # 내용 길이가 바뀌어도 좌측 패널 크기가 흔들리지 않게 고정
        right = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)

        # 왼쪽 패널 맨 아래: 마커 지우기/결과 복사/이미지 저장.
        # 아래쪽(side="bottom") 위젯은 가운데 목록(expand)보다 먼저 pack 해야 창이 작아져도 잘리지 않는다.
        bf = ctk.CTkFrame(left, fg_color="transparent")
        bf.pack(side="bottom", fill="x", padx=8, pady=(4, 8))
        for text, cmd in (("마커 지우기", self.clear_markers), ("결과 복사", self.copy_results),
                          ("이미지 저장", self.save_image)):
            self._btn(bf, text, cmd, fg_color=GRAY_BTN, hover_color="#4b5563", width=60).pack(
                side="left", expand=True, fill="x", padx=2)
        # '판독 옵션' 펼침 버튼 (옵션 내용은 아래 opt_frame 이며 평소에는 숨겨져 있다)
        self._opt_open = False
        self.btn_opt = ctk.CTkButton(left, text="⚙ 판독 옵션  ▾", command=self._toggle_options, height=26,
                                     font=self.f_s, fg_color="transparent", text_color="#374151",
                                     hover_color="#e5e7eb", border_width=1, border_color="#9ca3af")
        self.btn_opt.pack(side="bottom", fill="x", padx=10, pady=(4, 0))

        # 판독 옵션 내용 (pack 은 _toggle_options 에서 할 때만 수행)
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

        # 이미지 입력 (캡처 / 파일 열기 / 클립보드 붙여넣기)
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(8, 4))
        for text, cmd in (("캡처 F2", self.start_capture), ("열기", self.open_file), ("붙여넣기", self.paste)):
            self._btn(row, text, cmd, width=60).pack(side="left", expand=True, fill="x", padx=2)

        # 단위 / 구간 수 (구간 수 = 범례 색상 개수)
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        self._field(row, "단위", self.v_unit, width=52)
        ctk.CTkLabel(row, text="구간 수", font=self.f_s).pack(side="left")
        self.n_menu = ctk.CTkOptionMenu(row, values=[str(i) for i in range(2, 31)], width=56, height=24,
                                        font=self.f_s, dropdown_font=self.f_s, command=self._on_n_menu)
        self.n_menu.set(str(self.v_n.get()))
        self.n_menu.pack(side="left", padx=4)

        # 범례 값 / 색상 목록: 각 줄의 입력칸에 그 색 위쪽 경계의 값을 입력(비우면 자동 보간),
        # 맨 위 칸 = 최대값, 맨 아래 칸 = 최소값, 색상칸 클릭 = 색 수정
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

        # 오른쪽: 이미지 캔버스 (이미지/마커/라벨은 직접 그린다)
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

        # 하단 상태 표시줄 (조작 안내 / 진행 메시지)
        self.status = tk.StringVar(
            value="좌클릭: 마커 추가/선택 | 라벨 드래그: 위치 이동 | 휠: 확대/축소 | 우클릭 드래그: 이동 | Backspace: 선택 마커 삭제")
        ctk.CTkLabel(right, textvariable=self.status, font=ctk.CTkFont(family=FONT, size=11),
                     text_color="#4b5563", anchor="w").pack(fill="x", padx=6, pady=3)

        # 마우스 이벤트 연결: 이동=실시간 값, 좌클릭/드래그=마커·라벨, 우클릭 드래그=화면 이동, 휠=확대/축소
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
        """판독 옵션 패널을 펼치거나 접는다 (before=btn_opt 로 '옵션 버튼 바로 아래'에 위치시킨다)."""
        self._opt_open = not self._opt_open
        if self._opt_open:
            self.opt_frame.pack(side="bottom", fill="x", padx=8, pady=(2, 4), before=self.btn_opt)
        else:
            self.opt_frame.pack_forget()
        self.btn_opt.configure(text="⚙ 판독 옵션  ▴" if self._opt_open else "⚙ 판독 옵션  ▾")

    def _toggle_result(self):
        """판독 결과 상자의 자세한 내용(좌표/RGB/구간/보간 거리 등)을 펼치거나 접는다."""
        self._res_open = not self._res_open
        if self._res_open:
            self.res_info_lbl.pack(fill="x", padx=8, pady=(0, 6))
        else:
            self.res_info_lbl.pack_forget()
        self.btn_res.configure(text="▴" if self._res_open else "▾")

    def _on_n_menu(self, s: str):
        """구간 수 드롭다운 선택 처리."""
        self.v_n.set(int(s))
        self._on_n_change()

    def _bind_keys(self):
        """창 전체 단축키."""
        self.bind("<F2>", lambda e: self.start_capture())
        self.bind("<Control-o>", lambda e: self.open_file())
        self.bind("<Control-v>", lambda e: self.paste())
        self.bind("<BackSpace>", lambda e: self._pop_marker())
        self.bind("<Escape>", lambda e: self._cancel_legend_mode())

    # ------------------------------------------------------------ legend
    def _build_legend(self, quiet: bool = True) -> Legend | None:
        """현재 입력값으로 Legend 객체를 만든다. 입력이 올바르지 않으면 None
        (quiet=False 이면 오류 대화상자를 띄운다).
        마우스 이동마다 호출되므로 입력이 바뀌기 전까지는 만들어 둔 객체를 재사용한다."""
        if quiet and self._legend_cache is not None:
            return self._legend_cache
        try:
            n = len(self.colors)
            edges = {}
            if len(self.edge_vars) == n + 1:
                # 입력칸이 채워진 중간 경계값만 모은다 (빈칸은 자동 보간)
                for k in range(1, n):
                    s = self.edge_vars[k].get().strip()
                    if s:
                        edges[k] = float(s)
            self._legend_cache = Legend(
                float(self.v_min.get()), float(self.v_max.get()), self.colors,
                tolerance=self.v_tol.get(), open_ends=self.v_open.get(),
                unit=self.v_unit.get().strip(), ignore_black=self.v_black.get(),
                black_threshold=self.v_black_thr.get(), edges=edges, shade=self.v_shade.get())
            return self._legend_cache
        except (ValueError, tk.TclError) as e:
            if not quiet:
                messagebox.showerror("입력 오류", str(e))
            return None

    def _on_legend_change(self):
        """범례/옵션이 바뀌면 이전 판독 캐시를 버리고 목록과 화면을 다시 그린다."""
        self._legend_cache = None
        self._bmap_cache = None
        self._probe_cache.clear()
        self._refresh_swatches()
        self.render()

    def _on_n_change(self):
        """구간 수가 바뀌면 기존 색상들을 새 개수로 보간해서 채운다."""
        try:
            n = int(self.v_n.get())
        except (ValueError, tk.TclError):
            return
        if 2 <= n <= 30 and n != len(self.colors):
            self.colors = resample_colors(self.colors, n)
            self._on_legend_change()

    def _make_edge_vars(self, n: int, saved: dict | None = None):
        """경계값 입력 변수를 만든다. [0]=최소값(v_min), [n]=최대값(v_max)은 기존 변수를 그대로 쓰고,
        그 사이 경계는 새 변수를 만든다 (비워 두면 자동 보간). saved 로 저장된 값을 복원한다."""
        saved = saved or {}
        self.edge_vars = [self.v_min]
        for k in range(1, n):
            var = tk.StringVar(value=str(saved.get(str(k), saved.get(k, ""))))
            var.trace_add("write", lambda *_: self._on_legend_change())
            self.edge_vars.append(var)
        self.edge_vars.append(self.v_max)

    def clear_edges(self):
        """직접 입력한 중간 경계값을 모두 비운다 (최소/최대만 남겨 균등 분할)."""
        for var in self.edge_vars[1:-1]:
            var.set("")

    def _refresh_swatches(self):
        """색상 목록(경계값 입력칸 + 색상칸 + 구간 범위 문자열)을 갱신한다."""
        n = len(self.colors)
        rebuild = len(self._swatch_rows) != n
        if rebuild:   # 구간 수가 바뀔 때만 위젯을 다시 만든다 (깜빡임/흔들림 방지, 입력 중 커서 유지)
            self._make_edge_vars(n, self._saved_edges)
            self._saved_edges = None
            for w in self.swatch_frame.winfo_children():
                w.destroy()
            self._swatch_rows = []
            # 위 = 최대값. i 번 색의 입력칸은 그 색의 '위쪽 경계'(edge_vars[i + 1])에 해당한다
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
            # 맨 아래 줄: 최소값 입력칸
            row = ctk.CTkFrame(self.swatch_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkEntry(row, textvariable=self.edge_vars[0], width=64, height=22, font=self.f_s,
                         justify="right").pack(side="left")
            ctk.CTkLabel(row, text="← 최소", font=self.f_s, text_color="#9ca3af").pack(side="left", padx=6)
        # 색상과 구간 범위 문자열은 입력값이 바뀔 때마다 갱신
        lg = self._build_legend()
        for i, sw, t1 in self._swatch_rows:
            sw.configure(fg_color=self.colors[i])
            t1.configure(text=lg.short(i) if lg else "-")

    def _edit_color(self, i: int):
        """색상칸 클릭 시 색상 선택 대화상자로 i 번 구간의 색을 수정한다."""
        _, hx = colorchooser.askcolor(color=self.colors[i], parent=self, title=f"구간 {i + 1} 색상")
        if hx:
            self.colors[i] = hx
            self._on_legend_change()

    def reset_colors(self):
        """범례 색상을 기본 10단계로 되돌린다."""
        self.colors = list(DEFAULT_COLORS)
        self.v_n.set(len(self.colors))
        self.n_menu.set(str(len(self.colors)))
        self._on_legend_change()

    def toggle_legend_mode(self):
        """'범례 추출' 모드 전환. 켜면 이미지에서 범례 막대를 드래그해 색상을 자동으로 읽어온다."""
        if self.img is None:
            messagebox.showinfo("안내", "먼저 이미지를 가져오세요 (범례 막대가 포함된 이미지).")
            return
        self.legend_mode = not self.legend_mode
        self.btn_pick.configure(text="추출 취소(ESC)" if self.legend_mode else "범례 추출")
        self.status.set("범례 색상 막대를 드래그로 지정하세요 (세로: 위=최대, 가로: 오른쪽=최대)"
                        if self.legend_mode else "")

    def _cancel_legend_mode(self):
        """ESC: 범례 추출 모드 취소."""
        if self.legend_mode:
            self.toggle_legend_mode()

    # ----------------------------------------------------------- image in
    def set_image(self, img: Image.Image):
        """새 이미지를 불러온다. 이전 마커/캐시/결과를 모두 초기화하고 화면에 맞춰 표시한다."""
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
        """화면 캡처 시작: 창을 숨기고(지연 시간 + 0.25초 후) 캡처 오버레이를 띄운다."""
        try:
            delay = max(0, int(self.v_delay.get()))
        except (ValueError, tk.TclError):
            delay = 0
        self.withdraw()   # 이 프로그램 창이 캡처에 찍히지 않도록 숨긴다
        self.after(delay * 1000 + 250, self._do_capture)

    def _do_capture(self):
        """오버레이로 영역을 선택받아 이미지로 불러온다 (취소하면 아무 것도 하지 않음)."""
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
        """이미지 파일을 선택해서 불러온다."""
        p = filedialog.askopenfilename(
            parent=self, filetypes=[("이미지", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("모든 파일", "*.*")])
        if p:
            try:
                self.set_image(Image.open(p))
            except OSError as e:
                messagebox.showerror("열기 오류", str(e))

    def paste(self):
        """클립보드의 이미지(또는 복사한 이미지 파일)를 붙여넣는다."""
        data = ImageGrab.grabclipboard()
        if isinstance(data, Image.Image):
            self.set_image(data)
        elif isinstance(data, list) and data:   # 탐색기에서 파일을 복사한 경우 파일 경로 목록이 온다
            try:
                self.set_image(Image.open(data[0]))
            except OSError:
                messagebox.showinfo("안내", "클립보드에 이미지가 없습니다.")
        else:
            messagebox.showinfo("안내", "클립보드에 이미지가 없습니다.")

    # ---------------------------------------------------------- viewport
    # 좌표계: 이미지 좌표(px)와 캔버스 좌표를 (ox, oy, zoom) 으로 변환한다.
    #   캔버스 x = (이미지 x - ox) * zoom,   이미지 x = ox + 캔버스 x / zoom
    def fit(self):
        """이미지 전체가 캔버스에 들어오도록 배율/위치를 맞춘다 (작은 이미지는 확대하지 않음)."""
        if self.img is None:
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.zoom = min(cw / self.img.width, ch / self.img.height, 1.0)
        self.ox = (self.img.width - cw / self.zoom) / 2    # 이미지를 캔버스 중앙에 배치
        self.oy = (self.img.height - ch / self.zoom) / 2
        self.render()

    def _to_img(self, cx, cy):
        """캔버스 좌표 -> 이미지 좌표."""
        return self.ox + cx / self.zoom, self.oy + cy / self.zoom

    def _to_canvas(self, ix, iy):
        """이미지 좌표 -> 캔버스 좌표."""
        return (ix - self.ox) * self.zoom, (iy - self.oy) * self.zoom

    def render(self):
        """이미지와 마커를 다시 그린다. 화면에 보이는 부분만 잘라 확대/축소하므로 큰 이미지도 가볍다."""
        c = self.canvas
        c.delete("all")
        if self.img is None:
            c.create_text(c.winfo_width() // 2, c.winfo_height() // 2, fill="#aaa", font=(FONT, 14),
                          text="[캡처 F2] 로 판독할 영역을 드래그하거나\n열기 / 붙여넣기로 이미지를 가져오세요")
            return
        cw, ch, z = c.winfo_width(), c.winfo_height(), self.zoom
        # 캔버스에 보이는 이미지 영역 (이미지 범위로 제한)
        x0, y0 = max(0.0, self.ox), max(0.0, self.oy)
        x1, y1 = min(float(self.img.width), self.ox + cw / z), min(float(self.img.height), self.oy + ch / z)
        if x1 > x0 and y1 > y0:
            ix0, iy0, ix1, iy1 = math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)
            crop = self.img.crop((ix0, iy0, ix1, iy1))
            size = (max(1, round((ix1 - ix0) * z)), max(1, round((iy1 - iy0) * z)))
            # 확대할 때는 픽셀이 뭉개지지 않게 NEAREST, 축소할 때는 부드럽게 BILINEAR
            crop = crop.resize(size, Image.NEAREST if z >= 1 else Image.BILINEAR)
            self.photo = ImageTk.PhotoImage(crop)
            c.create_image((ix0 - self.ox) * z, (iy0 - self.oy) * z, anchor="nw", image=self.photo, tags="img")
        self.draw_overlay()

    def draw_overlay(self):
        """마커/라벨/연결선만 다시 그린다 (라벨 드래그 시 이미지는 그대로 두어 빠르게 갱신)."""
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
            sx, sy = self._to_canvas(m.x + 0.5, m.y + 0.5)   # 마커의 캔버스 위치 (픽셀 중앙)
            sel = i == self.selected
            border = self.colors[p.idx] if p and p.idx is not None else "#e00000"   # 라벨 테두리 = 판독된 구간 색
            short = p.label if (p and lg) else "-"
            u = unit if p and p.idx is not None else ""
            tag = f"lbl{i}"   # 라벨 클릭/드래그 판정에 쓰는 태그
            # 글자 크기를 재기 위해 임시로 그렸다가 지운다 (라벨 박스 크기 계산용)
            t = c.create_text(0, 0, text=f"#{i + 1}  {short}{u}", anchor="nw", font=(FONT, 11, "bold"))
            x0, y0, x1, y1 = c.bbox(t)
            c.delete(t)
            chip, pad = 14, 6   # 색상 칩 크기 / 안쪽 여백
            bw, bh = pad * 2 + chip + 6 + (x1 - x0), max(chip, y1 - y0) + pad * 2
            # 라벨 위치: 사용자가 옮긴 적이 없으면 자동(우상단, 화면 밖이면 반대쪽), 있으면 저장된 오프셋 사용
            if m.dx is None:
                dx, dy = auto_offset(bw, bh, cw, ch, sx, sy, 16)
            else:
                dx, dy = m.dx * self.zoom, m.dy * self.zoom
            bx, by = sx + dx, sy + dy
            self._label_boxes[i] = (bx, by, bw, bh)
            # 마커와 라벨 박스에서 가장 가까운 점을 잇는 연결선 (흰 테두리 + 어두운 선)
            nx, ny = min(max(sx, bx), bx + bw), min(max(sy, by), by + bh)
            if (nx, ny) != (sx, sy):
                c.create_line(sx, sy, nx, ny, fill="white", width=4, tags="ov")
                c.create_line(sx, sy, nx, ny, fill="#333333", width=2, tags="ov")
            # 마커: 원 + 십자선 (선택된 마커는 더 굵게)
            c.create_oval(sx - 7, sy - 7, sx + 7, sy + 7, outline="white", width=4, tags="ov")
            c.create_oval(sx - 7, sy - 7, sx + 7, sy + 7, outline="#ff1744" if sel else "#e00000", width=3 if sel else 2,
                          tags="ov")
            c.create_line(sx - 12, sy, sx + 12, sy, fill="#e00000", tags="ov")
            c.create_line(sx, sy - 12, sx, sy + 12, fill="#e00000", tags="ov")
            # 라벨 박스: 흰 바깥선 + 구간 색 테두리 + 색상 칩 + 값 문자열
            c.create_rectangle(bx - 2, by - 2, bx + bw + 2, by + bh + 2, fill="white", outline="white", tags=("ov", tag))
            c.create_rectangle(bx, by, bx + bw, by + bh, fill="white", outline=border, width=4 if sel else 3,
                               tags=("ov", tag))
            cy = by + bh / 2
            c.create_rectangle(bx + pad, cy - chip / 2, bx + pad + chip, cy + chip / 2, fill=border if p and p.idx is not None
                               else "#cccccc", outline="#555555", tags=("ov", tag))
            c.create_text(bx + pad + chip + 6, cy, text=f"#{i + 1}  {short}{u}", anchor="w",
                          font=(FONT, 11, "bold"), fill="#111111", tags=("ov", tag))

    def _tooltip(self, sx, sy, text, probe: Probe | None):
        """마우스 커서 옆에 따라다니는 값 말풍선. 화면 가장자리에서는 반대 방향으로 옮겨 그린다."""
        c = self.canvas
        border = self.colors[probe.idx] if probe and probe.idx is not None else "#e00000"
        t = c.create_text(sx + 14, sy - 14, text=text, anchor="sw", font=(FONT, 11, "bold"),
                          fill="black", tags="hover")
        x0, y0, x1, y1 = c.bbox(t)
        dx = dy = 0
        if x1 + 6 > c.winfo_width():   # 오른쪽 끝을 넘으면 커서 왼쪽으로
            dx = -(x1 - x0) - 28
        if y0 - 6 < 0:                 # 위쪽 끝을 넘으면 커서 아래쪽으로
            dy = (y1 - y0) + 28
        if dx or dy:
            c.move(t, dx, dy)
            x0, y0, x1, y1 = c.bbox(t)
        r = c.create_rectangle(x0 - 6, y0 - 4, x1 + 6, y1 + 4, fill="#fffde7", outline="white", width=5, tags="hover")
        r2 = c.create_rectangle(x0 - 6, y0 - 4, x1 + 6, y1 + 4, outline=border, width=3, tags="hover")
        c.tag_lower(r2, t)   # 배경 사각형들을 글자 아래로 내린다
        c.tag_lower(r, r2)

    # ------------------------------------------------------------- probe
    def _probe(self, ix: int, iy: int, lg: Legend | None) -> Probe | None:
        """이미지 좌표 (ix, iy) 를 판독한다. 같은 지점은 캐시를 사용한다 (범례/옵션이 바뀌면 캐시 초기화)."""
        if self.arr is None or lg is None:
            return None
        h, w = self.arr.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return None
        key = (ix, iy)
        if key in self._probe_cache:
            return self._probe_cache[key]
        r0 = int(self.v_sample.get()) // 2   # 샘플 영역 반경 (예: 샘플 5px -> 반경 2)
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
            refine_probe(p, self.arr, ix, iy, lg, radius, self._band_map(lg))   # 경계 거리 기반 소수 값 보간
        if len(self._probe_cache) > 5000:   # 캐시가 너무 커지지 않게 제한
            self._probe_cache.clear()
        self._probe_cache[key] = p
        return p

    def _band_map(self, lg: Legend) -> np.ndarray | None:
        """이미지 전체 구간 번호 캐시 (큰 이미지는 메모리 절약을 위해 None -> 클릭 주변만 계산)."""
        if self.arr.shape[0] * self.arr.shape[1] > 4_000_000:
            return None
        if self._bmap_cache is None:
            self._bmap_cache = lg.band_map(self.arr)
        return self._bmap_cache

    def _show_result(self, ix, iy, p: Probe | None):
        """우측 상단 결과 상자 갱신: 기본은 색+값, 펼치면 좌표/RGB/구간/일치도/보간 거리를 보여준다."""
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
        """구간 i 의 (하한, 상한) 을 표시용 문자열로 돌려준다."""
        lg = self._build_legend()
        lo, hi = lg.band_range(i) if lg else (0, 0)
        return fmt(lo), fmt(hi)

    # ------------------------------------------------------------ events
    def _on_motion(self, e):
        """마우스 이동: 커서 위치의 값을 실시간으로 판독해 결과 상자와 말풍선에 표시한다."""
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
        """좌클릭 시작: 라벨 위라면 라벨 드래그 준비, 아니면 클릭/패닝/범례 영역 선택의 시작점으로 기록한다."""
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
                # 마커 기준 라벨의 현재 오프셋을 기억해 두고 드래그한 만큼 더한다
                self._drag_label = (i, e.x, e.y, bx - sx, by - sy)
                if self.selected != i:
                    self.select(i)
                return

    def _on_drag(self, e):
        """좌클릭 드래그: 라벨 이동 / 범례 영역 선택 사각형 / 이미지 이동(패닝) 중 하나를 수행한다."""
        if not self._press:
            return
        sx, sy, ox, oy, moved = self._press
        if not moved and math.hypot(e.x - sx, e.y - sy) < 4:
            return   # 4px 미만은 클릭으로 간주 (손 떨림 방지)
        self._press = (sx, sy, ox, oy, True)
        if self._drag_label is not None:
            # 라벨 드래그: 마커 기준 오프셋을 이미지 좌표로 저장 (확대/축소해도 위치 유지)
            i, px, py, off_x, off_y = self._drag_label
            self.markers[i].dx = (off_x + e.x - px) / self.zoom
            self.markers[i].dy = (off_y + e.y - py) / self.zoom
            self.draw_overlay()
        elif self.legend_mode:
            # 범례 추출 모드: 드래그 영역을 점선 사각형으로 표시
            self.canvas.delete("legendbox")
            self.canvas.create_rectangle(sx, sy, e.x, e.y, outline="#00e5ff", width=2, dash=(4, 2), tags="legendbox")
        elif self.img is not None:
            # 일반 드래그: 이미지 이동
            self.ox = ox - (e.x - sx) / self.zoom
            self.oy = oy - (e.y - sy) / self.zoom
            self.render()

    def _on_release(self, e):
        """좌클릭 종료: 드래그가 아니었다면 마커를 선택하거나 새로 추가한다."""
        if not self._press or self.img is None:
            self._press = None
            return
        sx, sy, _, _, moved = self._press
        self._press = None
        if self._drag_label is not None:   # 라벨 드래그 종료
            self._drag_label = None
            return
        if self.legend_mode:               # 범례 영역 선택 종료 -> 색상 추출
            if moved:
                self._extract_legend(sx, sy, e.x, e.y)
            return
        if moved:                          # 이미지를 이동한 경우에는 마커를 찍지 않는다
            return
        hit = self._hit_marker(e.x, e.y)
        if hit is not None:                # 기존 마커를 클릭하면 선택만 한다
            self.select(hit)
            return
        fx, fy = self._to_img(e.x, e.y)
        ix, iy = int(math.floor(fx)), int(math.floor(fy))
        if 0 <= ix < self.img.width and 0 <= iy < self.img.height:
            self.markers.append(Marker(ix, iy))
            self.select(len(self.markers) - 1)

    def _hit_marker(self, cx, cy) -> int | None:
        """캔버스 좌표 근처(10px 이내)의 마커 번호. 겹치면 나중에 찍은 것이 우선."""
        for i in range(len(self.markers) - 1, -1, -1):
            sx, sy = self._to_canvas(self.markers[i].x + 0.5, self.markers[i].y + 0.5)
            if math.hypot(cx - sx, cy - sy) <= 10:
                return i
        return None

    def select(self, i: int | None):
        """마커를 선택하고 그 지점의 판독 결과를 결과 상자에 표시한다."""
        self.selected = i
        if i is not None:
            m = self.markers[i]
            self._show_result(m.x, m.y, self._probe(m.x, m.y, self._build_legend()))
        self.draw_overlay()

    def _extract_legend(self, cx0, cy0, cx1, cy1):
        """드래그한 캔버스 영역(범례 막대)에서 현재 구간 수만큼 색상을 추출해 범례 색으로 적용한다."""
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
        """우클릭 드래그(이미지 이동) 시작 위치 저장."""
        self._pan = (e.x, e.y, self.ox, self.oy)

    def _on_pan_move(self, e):
        """우클릭 드래그 중: 시작 위치 대비 이동량만큼 이미지를 옮긴다."""
        if self.img is None or self._pan is None:
            return
        sx, sy, ox, oy = self._pan
        self.ox = ox - (e.x - sx) / self.zoom
        self.oy = oy - (e.y - sy) / self.zoom
        self.render()

    def _on_wheel(self, e):
        """마우스 휠: 커서 위치를 기준으로 확대/축소 (0.05배 ~ 32배)."""
        if self.img is None:
            return
        fx, fy = self._to_img(e.x, e.y)   # 확대 전 커서가 가리키던 이미지 좌표
        z = self.zoom * (1.25 if e.delta > 0 else 0.8)
        self.zoom = min(32.0, max(0.05, z))
        self.ox = fx - e.x / self.zoom    # 같은 이미지 점이 커서 아래에 그대로 있도록 보정
        self.oy = fy - e.y / self.zoom
        self.render()

    # ----------------------------------------------------------- markers
    def clear_markers(self):
        """모든 마커 삭제."""
        self.markers.clear()
        self.select(None)

    def _pop_marker(self):
        """Backspace: 선택된 마커(없으면 마지막 마커)를 삭제. 입력칸에서 글자를 지우는 중에는 동작하지 않는다."""
        if self.markers and isinstance(self.focus_get(), (tk.Canvas, tk.Tk)):
            self.markers.pop(self.selected if self.selected is not None else -1)
            self.select(len(self.markers) - 1 if self.markers else None)

    def _marker_lines(self):
        """복사용 결과 목록: (번호, x, y, 값 문자열)."""
        lg = self._build_legend()
        out = []
        for n, m in enumerate(self.markers, 1):
            p = self._probe(m.x, m.y, lg)
            out.append((n, m.x, m.y, p.text if p else "-"))
        return out

    def copy_results(self):
        """모든 마커의 판독 값을 탭으로 구분해 클립보드에 복사한다 (엑셀에 바로 붙여넣기 가능)."""
        rows = self._marker_lines()
        if not rows:
            return
        self.clipboard_clear()
        self.clipboard_append("\n".join(f"#{n}\t({x},{y})\t{t}" for n, x, y, t in rows))
        self.status.set(f"마커 {len(rows)}개 결과를 클립보드에 복사했습니다.")

    def save_image(self):
        """마커와 라벨이 표시된 이미지를 PNG 로 저장한다 (라벨 위치/색상은 화면과 동일)."""
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
        """창을 닫을 때 현재 입력값(범례, 옵션, 중간 경계값)을 설정 파일에 저장한다."""
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
            pass   # 저장 실패(쓰기 권한 없음 등)는 무시하고 그냥 종료한다
        self.destroy()


def main():
    """프로그램 시작점."""
    App().mainloop()
