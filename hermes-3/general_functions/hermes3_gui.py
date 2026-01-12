"""
Standalone Hermes-3 1D GUI (Matplotlib widgets; **no Tkinter**).

This avoids the common macOS/Homebrew situation where Python lacks `_tkinter`.

Run from terminal:

```bash
python hermes3_gui.py /path/to/case_dir
```

If no path is given, the app opens with no data loaded; enter a case directory
path in the top-left text box and click "Load dataset".
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

if platform.system() == "Darwin":
    # Explicitly avoid TkAgg; Homebrew Python often lacks `_tkinter`.
    matplotlib.use("MacOSX")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button, Slider, TextBox  # noqa: E402


def _infer_time_dim(ds) -> Optional[str]:
    for cand in ("t", "time"):
        if cand in ds.dims:
            return cand
    # fall back: any 1D dimension with monotonic coordinate
    for d in ds.dims:
        if d in ds.coords and ds[d].ndim == 1:
            return d
    return None


def _infer_spatial_dim(ds) -> str:
    # Hermes-3 1D typically uses "pos"
    for cand in ("pos", "y", "x", "s"):
        if cand in ds.dims:
            return cand
    # fall back: choose a non-time dimension
    tdim = _infer_time_dim(ds)
    for d in ds.dims:
        if d != tdim:
            return d
    # last resort
    return list(ds.dims)[0]


def _is_plottable_1d_var(da, spatial_dim: str, time_dim: Optional[str]) -> bool:
    dims = tuple(da.dims)
    if spatial_dim not in dims:
        return False
    if len(dims) == 1 and dims[0] == spatial_dim:
        return True
    if time_dim is None:
        return False
    if len(dims) == 2 and set(dims) == {time_dim, spatial_dim}:
        return True
    return False


def _list_plottable_vars(ds, spatial_dim: str, time_dim: Optional[str]) -> List[str]:
    out: List[str] = []
    for name, da in ds.data_vars.items():
        try:
            if _is_plottable_1d_var(da, spatial_dim=spatial_dim, time_dim=time_dim):
                out.append(name)
        except Exception:
            continue
    return sorted(out)


def _format_case_label(case_path: str) -> str:
    p = Path(case_path).expanduser().resolve()
    return p.name or str(p)

def _truncate_middle(s: str, max_len: int) -> str:
    """
    Truncate a long string with an ellipsis in the middle to keep the start/end visible.
    """
    s = str(s)
    if max_len <= 0 or len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    keep = max_len - 1  # account for ellipsis char
    left = keep // 2
    right = keep - left
    return s[:left] + "…" + s[-right:]


@dataclass
class _LoadedCase:
    label: str
    case_path: str
    ds: "object"  # xarray.Dataset (kept generic to avoid hard dependency at import time)
    n_time: int = 1  # number of time steps in this dataset

def _ensure_sdtools_on_path():
    """
    Make a best-effort attempt to ensure `analysis/sdtools` is importable.

    This script lives in: analysis/notebooks/hermes-3/general_functions/
    sdtools lives in:        analysis/sdtools/
    """
    here = Path(__file__).resolve()
    # Walk upwards until we find an `analysis/sdtools` directory.
    # This avoids hardcoding fragile parent indices.
    for parent in [here.parent, *here.parents]:
        sdtools_dir = parent / "analysis" / "sdtools"
        if sdtools_dir.exists():
            sp = str(sdtools_dir)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return

    # Fallback: if the script itself is inside `.../analysis/...`, locate that `analysis/`.
    for parent in [here.parent, *here.parents]:
        if parent.name == "analysis":
            sdtools_dir = parent / "sdtools"
            if sdtools_dir.exists():
                sp = str(sdtools_dir)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
            return


class Hermes3GuiApp:
    """
    Matplotlib-window GUI driven by matplotlib.widgets.

    Because there is no native dropdown in matplotlib.widgets, we use a TextBox per subplot
    for the variable name. Available variables are listed in the left panel for reference.
    """

    def __init__(self, *, initial_case_path: Optional[str], n_plots: int, spatial_dim: Optional[str]):
        _ensure_sdtools_on_path()
        try:
            from hermes3.load import Load  # type: ignore
        except Exception as e:
            raise ImportError(
                "Could not import `hermes3.load.Load`.\n"
                "Fix by either:\n"
                "- setting PYTHONPATH to include `/Users/lloyd/Documents/hermes_dir/analysis/sdtools`, or\n"
                "- running this script from within the repo where `analysis/sdtools` exists.\n"
                f"Original error: {e}"
            ) from e

        self.Load = Load
        self.cases: Dict[str, _LoadedCase] = {}
        self.spatial_dim_forced = spatial_dim
        self.state = dict(spatial_dim=None, time_dim=None, vars=[], t_values=None)

        # `n_plots` kept for CLI backward-compat but plots are now driven by toggled vars.
        self.n_plots = max(1, int(n_plots))
        self.selected_vars: List[str] = []  # preserve selection order
        self._selected_set = set()
        self._var_scroll = 0
        self._n_visible_vars = 18
        self._var_line_artists: List[Tuple[str, "object"]] = []  # (varname, Text)
        self._cid_click = None
        self._cid_scroll = None
        # Per-variable y-scale state
        self._yscale_by_var: Dict[str, str] = {}  # var -> {"linear","log","symlog"}
        self._scale_button_axes: List["plt.Axes"] = []
        self._scale_buttons: Dict[str, Button] = {}
        # Per-variable y-limit mode state
        self._ylim_mode_by_var: Dict[str, str] = {}  # var -> {"auto","final","global"}
        self._ylim_button_axes: List["plt.Axes"] = []
        self._ylim_buttons: Dict[str, Button] = {}
        self._max_var_label_chars = 34
        self._var_filter = ""
        self._clipboard_hint_shown = False

        self._build_figure(initial_case_path=initial_case_path)

    # ---------- Figure layout ----------
    def _build_figure(self, *, initial_case_path: Optional[str]):
        self.fig = plt.figure(figsize=(12.5, 7.5))
        self.fig.canvas.manager.set_window_title("Hermes-3 GUI (1D)")

        # Layout constants in figure coords
        left_x0, left_w = 0.04, 0.26
        right_x0, right_w = 0.34, 0.62
        top_y0, top_y1 = 0.78, 0.95
        plots_y0, plots_y1 = 0.18, 0.95
        slider_y0, slider_h = 0.08, 0.04
        gap = 0.04  # Increased gap between subplots to prevent overlap

        # Dataset path textbox (top-left)
        ax_path = self.fig.add_axes([left_x0, top_y1 - 0.06, left_w * 0.70, 0.045])
        self.path_box = TextBox(ax_path, "dataset path", initial=(initial_case_path or ""))
        # Enable text clipping to prevent long paths from overlapping other elements
        self.path_box.text_disp.set_clip_on(True)
        ax_copy = self.fig.add_axes([left_x0 + left_w * 0.72, top_y1 - 0.06, left_w * 0.13, 0.045])
        ax_paste = self.fig.add_axes([left_x0 + left_w * 0.87, top_y1 - 0.06, left_w * 0.13, 0.045])
        self.copy_path_btn = Button(ax_copy, "Copy")
        self.paste_path_btn = Button(ax_paste, "Paste")

        # Buttons row
        ax_load = self.fig.add_axes([left_x0, top_y1 - 0.115, left_w * 0.48, 0.04])
        ax_add = self.fig.add_axes([left_x0 + left_w * 0.52, top_y1 - 0.115, left_w * 0.48, 0.04])
        self.load_btn = Button(ax_load, "Load dataset")
        self.add_btn = Button(ax_add, "Load additional")

        # Status panel (axes so it can be clipped/wrapped without spilling into plots)
        ax_status = self.fig.add_axes([left_x0, top_y1 - 0.205, left_w, 0.07])
        ax_status.set_axis_off()
        self._status_ax = ax_status
        self.status_text = ax_status.text(
            0.0,
            1.0,
            "",
            fontsize=10,
            color="#333",
            va="top",
            ha="left",
            wrap=True,
            clip_on=True,
            transform=ax_status.transAxes,
        )

        # Loaded datasets list (positioned below status panel, above search box)
        # Status ends at 0.745, so start datasets at 0.72 with small height
        ax_datasets = self.fig.add_axes([left_x0, 0.695, left_w, 0.045])
        ax_datasets.set_axis_off()
        self._datasets_ax = ax_datasets
        self._datasets_text = ax_datasets.text(
            0.0,
            1.0,
            "",
            fontsize=8,
            color="#333",
            va="top",
            ha="left",
            family="monospace",
            wrap=True,
            clip_on=True,
            transform=ax_datasets.transAxes,
        )

        # Variable entry boxes (left column)
        # Variable list (scrollable + clickable)
        # Search box
        ax_search = self.fig.add_axes([left_x0, 0.645, left_w, 0.04])
        self.search_box = TextBox(ax_search, "search", initial="")

        ax_list_title = self.fig.add_axes([left_x0, 0.595, left_w, 0.035])
        ax_list_title.set_axis_off()
        ax_list_title.text(0.0, 0.0, "Variables (click to toggle)", fontsize=10, color="#333", va="bottom", ha="left", clip_on=True, transform=ax_list_title.transAxes)

        ax_list = self.fig.add_axes([left_x0, 0.20, left_w, 0.39])
        ax_list.set_axis_off()
        self.var_list_ax = ax_list

        # Deselect all button
        ax_deselect = self.fig.add_axes([left_x0, 0.155, left_w, 0.04])
        self.deselect_btn = Button(ax_deselect, "Deselect All")

        self._render_var_list()

        # Plot axes (right column) are dynamic based on selection
        self._plot_area = dict(right_x0=right_x0, right_w=right_w, plots_y0=plots_y0, plots_y1=plots_y1, gap=gap)
        self.axes: List["plt.Axes"] = []
        self._rebuild_plot_axes()

        # Time slider (bottom, under plots)
        ax_slider = self.fig.add_axes([right_x0, slider_y0, right_w, slider_h])
        self.time_slider = Slider(ax_slider, "time index", valmin=0, valmax=0, valinit=0, valstep=1)
        self.time_readout = self.fig.text(right_x0, slider_y0 + slider_h + 0.01, "", fontsize=10, color="#333")

        # Wire callbacks
        self.load_btn.on_clicked(lambda _evt: self.load_dataset(replace=True))
        self.add_btn.on_clicked(lambda _evt: self.load_dataset(replace=False))
        self.time_slider.on_changed(lambda _val: self.redraw())
        self.copy_path_btn.on_clicked(lambda _evt: self.copy_dataset_path())
        self.paste_path_btn.on_clicked(lambda _evt: self.paste_dataset_path())
        self.deselect_btn.on_clicked(lambda _evt: self.deselect_all_vars())

        # Click + scroll handlers for variable list, double-click for path box
        self._cid_click = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._cid_scroll = self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._cid_dblclick = self.fig.canvas.mpl_connect("button_press_event", self._on_double_click)
        # Search box live filtering (matplotlib>=3.8 has on_text_change)
        try:
            self.search_box.on_text_change(self._on_search_change)  # type: ignore[attr-defined]
        except Exception:
            # Fallback: filter on Enter only
            self.search_box.on_submit(self._on_search_change)

        # Start state
        self._update_datasets_list()  # Initialize with empty list
        if initial_case_path:
            self.load_dataset(replace=True)
        else:
            self.set_status("Enter a case directory path and click 'Load dataset'.")
            self.redraw()

    def _get_clipboard_text(self) -> str:
        """
        Best-effort clipboard read without extra deps.
        macOS: pbpaste
        Linux: xclip or wl-paste
        """
        if platform.system() == "Darwin":
            return subprocess.check_output(["pbpaste"], text=True)
        # Linux fallbacks
        for cmd in (["wl-paste"], ["xclip", "-selection", "clipboard", "-o"]):
            try:
                return subprocess.check_output(cmd, text=True)
            except Exception:
                continue
        raise RuntimeError("No clipboard command found (need pbpaste/wl-paste/xclip).")

    def _set_clipboard_text(self, text: str) -> None:
        """
        Best-effort clipboard write without extra deps.
        macOS: pbcopy
        Linux: xclip or wl-copy
        """
        if platform.system() == "Darwin":
            subprocess.run(["pbcopy"], input=text, text=True, check=True)
            return
        for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"]):
            try:
                subprocess.run(cmd, input=text, text=True, check=True)
                return
            except Exception:
                continue
        raise RuntimeError("No clipboard command found (need pbcopy/wl-copy/xclip).")

    def copy_dataset_path(self):
        try:
            self._set_clipboard_text(self.path_box.text or "")
            self.set_status("Copied dataset path to clipboard.")
        except Exception as e:
            self.set_status(f"Copy failed: {e}", is_error=True)

    def paste_dataset_path(self):
        try:
            txt = self._get_clipboard_text().strip()
            if txt:
                self.path_box.set_val(txt)
                self.set_status("Pasted dataset path from clipboard.")
            else:
                self.set_status("Clipboard is empty.", is_error=True)
        except Exception as e:
            self.set_status(f"Paste failed: {e}", is_error=True)
    def _set_var_list_message(self, msg: str):
        self.var_list_ax.clear()
        self.var_list_ax.set_axis_off()
        self._var_line_artists = []
        self.var_list_ax.text(
            0.0,
            1.0,
            msg,
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            transform=self.var_list_ax.transAxes,
            clip_on=True,
        )

    def _filtered_vars(self) -> List[str]:
        vars_all = list(self.state.get("vars") or [])
        q = (self._var_filter or "").strip().lower()
        if not q:
            return vars_all
        return [v for v in vars_all if q in v.lower()]

    def _on_search_change(self, text: str):
        self._var_filter = text or ""
        self._var_scroll = 0
        self._render_var_list()
        self.fig.canvas.draw_idle()

    def _render_var_list(self):
        """
        Render a scrollable list of variables in `self.var_list_ax`.
        Clicking a line toggles selection.
        """
        vars_all = list(self.state.get("vars") or [])
        vars_ = self._filtered_vars()
        if not vars_all:
            self._set_var_list_message("No dataset loaded.")
            return
        if vars_all and not vars_:
            self._set_var_list_message("No matches.")
            return

        # Clamp scroll
        max_scroll = max(0, len(vars_) - self._n_visible_vars)
        self._var_scroll = max(0, min(int(self._var_scroll), max_scroll))
        visible = vars_[self._var_scroll : self._var_scroll + self._n_visible_vars]

        self.var_list_ax.clear()
        self.var_list_ax.set_axis_off()
        self._var_line_artists = []

        # Header with counts
        sel_n = len(self.selected_vars)
        header = (
            f"Showing {self._var_scroll+1}-{min(self._var_scroll+self._n_visible_vars, len(vars_))} / {len(vars_)}"
            f"   (of {len(vars_all)})   |   selected: {sel_n}"
        )
        self.var_list_ax.text(
            0.0,
            1.0,
            header,
            va="top",
            ha="left",
            fontsize=9,
            color="#444",
            transform=self.var_list_ax.transAxes,
            clip_on=True,
        )

        # Render each line
        line_h = 1.0 / (self._n_visible_vars + 1)
        y = 1.0 - line_h * 1.5  # leave room for header
        for name in visible:
            selected = name in self._selected_set
            prefix = "[x] " if selected else "[ ] "
            color = "#0b6" if selected else "#111"
            disp = prefix + _truncate_middle(name, self._max_var_label_chars)
            txt = self.var_list_ax.text(
                0.0,
                y,
                disp,
                va="center",
                ha="left",
                fontsize=9,
                family="monospace",
                color=color,
                transform=self.var_list_ax.transAxes,
                clip_on=True,
            )
            self._var_line_artists.append((name, txt))
            y -= line_h

    def _toggle_var(self, name: str):
        if name in self._selected_set:
            self._selected_set.remove(name)
            self.selected_vars = [v for v in self.selected_vars if v != name]
        else:
            self._selected_set.add(name)
            self.selected_vars.append(name)
            self._yscale_by_var.setdefault(name, "linear")
            self._ylim_mode_by_var.setdefault(name, "auto")

        self._rebuild_plot_axes()
        self._render_var_list()
        self.redraw()

    def deselect_all_vars(self):
        """Deselect all currently selected variables."""
        self.selected_vars = []
        self._selected_set = set()
        self._rebuild_plot_axes()
        self._render_var_list()
        self.redraw()

    def _cycle_yscale(self, current: str) -> str:
        order = ["linear", "log", "symlog"]
        try:
            i = order.index(current)
        except ValueError:
            return "linear"
        return order[(i + 1) % len(order)]

    def _yscale_label(self, mode: str) -> str:
        if mode == "log":
            return "y:log"
        if mode == "symlog":
            return "y:symlog"
        return "y:lin"

    def _cycle_ylim_mode(self, current: str) -> str:
        order = ["auto", "final", "global"]
        try:
            i = order.index(current)
        except ValueError:
            return "auto"
        return order[(i + 1) % len(order)]

    def _ylim_mode_label(self, mode: str) -> str:
        if mode == "final":
            return "ylim:final"
        if mode == "global":
            return "ylim:max"
        return "ylim:auto"

    def _clear_scale_buttons(self):
        # Remove axes that hosted buttons
        for axb in list(self._scale_button_axes):
            try:
                axb.remove()
            except Exception:
                pass
        self._scale_button_axes = []
        self._scale_buttons = {}

    def _clear_ylim_buttons(self):
        # Remove axes that hosted buttons
        for axb in list(self._ylim_button_axes):
            try:
                axb.remove()
            except Exception:
                pass
        self._ylim_button_axes = []
        self._ylim_buttons = {}

    def _rebuild_scale_buttons(self):
        """
        Create one y-scale toggle button per active subplot, positioned at top-right of each axes.
        """
        self._clear_scale_buttons()

        vars_to_plot = list(self.selected_vars)
        if not vars_to_plot:
            return

        # Button size in figure coordinates (tuned for readability)
        w, h = 0.06, 0.032
        pad = 0.004

        for ax, var in zip(self.axes, vars_to_plot):
            pos = ax.get_position()  # in figure coords
            x0 = pos.x1 - w - pad
            y0 = pos.y1 - h - pad
            axb = self.fig.add_axes([x0, y0, w, h])
            mode = self._yscale_by_var.get(var, "linear")
            b = Button(axb, self._yscale_label(mode))

            def _make_cb(vname: str):
                def _cb(_evt):
                    cur = self._yscale_by_var.get(vname, "linear")
                    nxt = self._cycle_yscale(cur)
                    self._yscale_by_var[vname] = nxt
                    # Update label immediately
                    btn = self._scale_buttons.get(vname)
                    if btn is not None:
                        btn.label.set_text(self._yscale_label(nxt))
                    self.redraw()
                return _cb

            b.on_clicked(_make_cb(var))
            self._scale_button_axes.append(axb)
            self._scale_buttons[var] = b

    def _rebuild_ylim_buttons(self):
        """
        Create one y-limit mode toggle button per active subplot, positioned next to y-scale button.
        """
        self._clear_ylim_buttons()

        vars_to_plot = list(self.selected_vars)
        if not vars_to_plot:
            return

        # Button size in figure coordinates (wider to fit text)
        w_ylim = 0.08
        h = 0.032
        pad = 0.004
        gap = 0.003  # gap between ylim and yscale buttons
        w_yscale = 0.06  # width of yscale button

        for ax, var in zip(self.axes, vars_to_plot):
            pos = ax.get_position()  # in figure coords
            # Position to the left of the yscale button
            x0 = pos.x1 - w_yscale - pad - gap - w_ylim
            y0 = pos.y1 - h - pad
            axb = self.fig.add_axes([x0, y0, w_ylim, h])
            mode = self._ylim_mode_by_var.get(var, "auto")
            b = Button(axb, self._ylim_mode_label(mode))

            def _make_cb(vname: str):
                def _cb(_evt):
                    cur = self._ylim_mode_by_var.get(vname, "auto")
                    nxt = self._cycle_ylim_mode(cur)
                    self._ylim_mode_by_var[vname] = nxt
                    # Update label immediately
                    btn = self._ylim_buttons.get(vname)
                    if btn is not None:
                        btn.label.set_text(self._ylim_mode_label(nxt))
                    self.redraw()
                return _cb

            b.on_clicked(_make_cb(var))
            self._ylim_button_axes.append(axb)
            self._ylim_buttons[var] = b

    def _on_double_click(self, event):
        """Clear the dataset path textbox on double-click for easy replacement."""
        if event.dblclick and event.inaxes == self.path_box.ax:
            self.path_box.set_val("")
            self.set_status("Path cleared. Enter new dataset path.")

    def _on_click(self, event):
        # Only respond to clicks inside the variable list axes
        if event.inaxes != self.var_list_ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Convert y in axes coords to which line was clicked
        # event.ydata is in data coords; since we use transAxes, map using transforms.
        inv = self.var_list_ax.transAxes.inverted()
        _, y_ax = inv.transform((event.x, event.y))
        if y_ax < 0 or y_ax > 1:
            return

        # Determine index (must match `_render_var_list` layout):
        # - header is drawn at y=1.0 (top)
        # - first line center is at y0 = 1.0 - 1.5*line_h
        line_h = 1.0 / (self._n_visible_vars + 1)
        y0 = 1.0 - line_h * 1.5
        # Nearest line index to click position
        idx = int(np.floor((y0 - y_ax) / line_h + 0.5))
        if idx < 0 or idx >= len(self._var_line_artists):
            return
        name = self._var_line_artists[idx][0]
        self._toggle_var(name)

    def _on_scroll(self, event):
        if event.inaxes != self.var_list_ax:
            return
        vars_ = self._filtered_vars()
        if not vars_:
            return
        max_scroll = max(0, len(vars_) - self._n_visible_vars)

        # Matplotlib reports scroll step with sign; fall back to button names if needed
        step = getattr(event, "step", None)
        if step is None:
            # some backends provide event.button = 'up'/'down'
            step = 1 if getattr(event, "button", "") == "up" else -1
        self._var_scroll -= int(step)
        self._var_scroll = max(0, min(self._var_scroll, max_scroll))
        self._render_var_list()
        self.fig.canvas.draw_idle()

    def _rebuild_plot_axes(self):
        # Remove existing plot axes
        for ax in list(self.axes):
            try:
                ax.remove()
            except Exception:
                pass
        self.axes = []

        # Build new axes based on selection (at least 1 for empty state)
        sel = list(self.selected_vars)
        n = max(1, len(sel))
        right_x0 = self._plot_area["right_x0"]
        right_w = self._plot_area["right_w"]
        plots_y0 = self._plot_area["plots_y0"]
        plots_y1 = self._plot_area["plots_y1"]
        gap = self._plot_area["gap"]

        # Layout: Maximum 3 rows, add columns as needed
        nrows = min(3, n)  # max 3 rows
        ncols = int(np.ceil(n / nrows))
        self._last_plot_grid = (nrows, ncols)  # for labeling logic in redraw()

        total_h = plots_y1 - plots_y0
        per_h = (total_h - gap * (nrows - 1)) / nrows

        # Horizontal spacing between columns (increased to prevent y-label overlap)
        col_gap = 0.08
        col_w = (right_w - col_gap * (ncols - 1)) / ncols

        # sharex within each column
        sharex_ref: List[Optional["plt.Axes"]] = [None] * ncols

        # Build in selection order: fill down first column, then next column
        for idx in range(n):
            col = idx // nrows
            row = idx % nrows
            x0 = right_x0 + col * (col_w + col_gap)
            y0 = plots_y1 - (row + 1) * per_h - row * gap
            ax = self.fig.add_axes([x0, y0, col_w, per_h], sharex=sharex_ref[col])
            if sharex_ref[col] is None:
                sharex_ref[col] = ax
            self.axes.append(ax)

        # Hide x tick labels except for the bottom-most axis in each column
        bottom_idx_by_col: Dict[int, int] = {}
        for col in range(ncols):
            inds = [i for i in range(n) if (i // nrows) == col]
            if inds:
                bottom_idx_by_col[col] = max(inds)

        for i, ax in enumerate(self.axes):
            col = i // nrows
            is_bottom = bottom_idx_by_col.get(col, -1) == i
            if not is_bottom:
                ax.tick_params(labelbottom=False)

        # Buttons depend on axes positions
        self._rebuild_scale_buttons()
        self._rebuild_ylim_buttons()

    # ---------- Data ----------
    def set_status(self, msg: str, *, is_error: bool = False):
        self.status_text.set_text(msg)
        self.status_text.set_color("#b00020" if is_error else "#333")
        self.fig.canvas.draw_idle()

    def _update_datasets_list(self):
        """Update the display of loaded datasets."""
        if not self.cases:
            self._datasets_text.set_text("Loaded datasets: (none)")
        else:
            labels = [c.label for c in self.cases.values()]
            # Show only first 2 datasets, then indicate if there are more
            if len(labels) <= 2:
                header = f"Loaded datasets ({len(labels)}):"
                items = "\n".join(f"  • {lbl}" for lbl in labels)
                self._datasets_text.set_text(f"{header}\n{items}")
            else:
                # Show first 2 and indicate there are more
                header = f"Loaded datasets ({len(labels)}):"
                shown = "\n".join(f"  • {lbl}" for lbl in labels[:2])
                more = f"  ... and {len(labels) - 2} more"
                self._datasets_text.set_text(f"{header}\n{shown}\n{more}")
        self.fig.canvas.draw_idle()

    def _load_case(self, case_path: str) -> _LoadedCase:
        case_path = str(Path(case_path).expanduser().resolve())
        label = _format_case_label(case_path)
        cs = self.Load.case_1D(case_path, verbose=False)
        # Determine number of time steps
        tdim = _infer_time_dim(cs.ds)
        n_time = int(cs.ds.sizes[tdim]) if tdim and tdim in cs.ds.dims else 1
        return _LoadedCase(label=label, case_path=case_path, ds=cs.ds, n_time=n_time)

    def _recompute_common_vars(self) -> Tuple[List[str], Optional[str], Optional[str]]:
        if not self.cases:
            return [], None, None
        first = next(iter(self.cases.values())).ds
        tdim = _infer_time_dim(first)
        sdim = self.spatial_dim_forced or _infer_spatial_dim(first)

        # Use union instead of intersection - show all variables from any dataset
        all_vars = set(_list_plottable_vars(first, spatial_dim=sdim, time_dim=tdim))
        for c in list(self.cases.values())[1:]:
            all_vars |= set(_list_plottable_vars(c.ds, spatial_dim=sdim, time_dim=tdim))
        return sorted(all_vars), sdim, tdim

    def _update_after_load(self):
        vars_, sdim, tdim = self._recompute_common_vars()
        self.state["vars"] = vars_
        self.state["spatial_dim"] = sdim
        self.state["time_dim"] = tdim

        # Drop selections that no longer exist
        if vars_:
            keep = [v for v in self.selected_vars if v in vars_]
            self.selected_vars = keep
            self._selected_set = set(keep)
            # If nothing selected yet, select Te by default (or first var if Te not available)
            if not self.selected_vars:
                default_var = "Te" if "Te" in vars_ else vars_[0]
                self.selected_vars = [default_var]
                self._selected_set = {default_var}
        else:
            self.selected_vars = []
            self._selected_set = set()
        self._var_scroll = 0
        self._render_var_list()
        self._rebuild_plot_axes()

        # Update slider range based on maximum time steps across all cases
        max_n_t = max((c.n_time for c in self.cases.values()), default=1)

        # Get time values from first dataset for display
        ds0 = next(iter(self.cases.values())).ds
        t_values = None
        if tdim is not None and tdim in ds0.coords:
            try:
                t_values = np.asarray(ds0[tdim].values)
            except Exception:
                t_values = None
        self.state["t_values"] = t_values
        self._set_time_range(max(1, max_n_t))

    def _set_time_range(self, n_t: int):
        n_t = max(1, int(n_t))
        self.time_slider.valmin = 0
        self.time_slider.valmax = max(0, n_t - 1)
        self.time_slider.ax.set_xlim(self.time_slider.valmin, self.time_slider.valmax)
        # Set to final time step by default
        self.time_slider.set_val(n_t - 1)

    def _get_time_index(self) -> int:
        try:
            return int(self.time_slider.val)
        except Exception:
            return 0

    def _get_time_index_for_case(self, case: _LoadedCase) -> int:
        """
        Get the time index for a specific case, clamping to its available range.
        """
        ti = self._get_time_index()
        return min(ti, case.n_time - 1)

    # ---------- Actions ----------
    def load_dataset(self, *, replace: bool):
        p = (self.path_box.text or "").strip()
        if not p:
            self.set_status("Please enter a case directory path.", is_error=True)
            return
        try:
            lc = self._load_case(p)
            if replace:
                self.cases.clear()
            self.cases[lc.label] = lc
            self._update_after_load()
            self._update_datasets_list()
            self.set_status("")  # Clear status message, datasets list shows what's loaded
            self.redraw()
        except Exception as e:
            self.set_status(f"Failed to load dataset: {e}", is_error=True)

    # ---------- Plotting ----------
    def _compute_ylim_for_final(self, varname: str, tdim: Optional[str], yscale: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute y-limits based on data at the final time step across all cases.
        """
        ys_all = []
        for c in self.cases.values():
            ds = c.ds
            if varname not in ds:
                continue
            da = ds[varname]
            try:
                # Get final time index for this case
                final_ti = c.n_time - 1
                if tdim is not None and tdim in da.dims:
                    da1 = da.isel({tdim: final_ti})
                else:
                    da1 = da
                yv = np.asarray(da1.values)
                yv = yv[np.isfinite(yv)]
                if yscale == "log":
                    yv = yv[yv > 0]  # filter non-positive for log scale
                if yv.size:
                    ys_all.append(yv)
            except Exception:
                continue
        if not ys_all:
            return None, None
        ys = np.concatenate(ys_all)
        if ys.size == 0:
            return None, None
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        # Add 5% margin
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.1 * abs(ymax)
        return ymin - margin, ymax + margin

    def _compute_ylim_for_global(self, varname: str, yscale: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute y-limits based on global min/max across all time steps and cases.
        """
        ys_all = []
        for c in self.cases.values():
            ds = c.ds
            if varname not in ds:
                continue
            da = ds[varname]
            try:
                yv = np.asarray(da.values)
                yv = yv[np.isfinite(yv)]
                if yscale == "log":
                    yv = yv[yv > 0]  # filter non-positive for log scale
                if yv.size:
                    ys_all.append(yv)
            except Exception:
                continue
        if not ys_all:
            return None, None
        ys = np.concatenate(ys_all)
        if ys.size == 0:
            return None, None
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        # Add 5% margin
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.1 * abs(ymax)
        return ymin - margin, ymax + margin

    def _update_time_readout(self):
        ti = self._get_time_index()
        tdim = self.state.get("time_dim")
        tvals = self.state.get("t_values")

        # Show time value from first dataset if available
        if tdim and tvals is not None and ti < len(tvals):
            self.time_readout.set_text(f"{tdim} = {tvals[ti] * 1e3:.4f} ms")
        else:
            self.time_readout.set_text(f"time index = {ti}")

    def redraw(self):
        self._update_time_readout()
        for ax in self.axes:
            ax.clear()

        if not self.cases:
            self.fig.canvas.draw_idle()
            return

        sdim = self.state["spatial_dim"]
        tdim = self.state["time_dim"]
        vars_to_plot = list(self.selected_vars)
        if not vars_to_plot:
            # Empty state: show message on first axis
            ax0 = self.axes[0]
            ax0.set_axis_off()
            ax0.text(0.5, 0.5, "No variables selected.\nClick variables on the left to toggle plots.",
                     ha="center", va="center", fontsize=12, transform=ax0.transAxes)
            self.fig.canvas.draw_idle()
            return

        # Ensure axes count matches selection (can happen if selection changed very fast)
        if len(self.axes) != len(vars_to_plot):
            self._rebuild_plot_axes()

        for ax, name in zip(self.axes, vars_to_plot):
            ax.set_axis_on()
            mode = self._yscale_by_var.get(name, "linear")
            # Ensure button labels reflect current state
            btn = self._scale_buttons.get(name)
            if btn is not None:
                btn.label.set_text(self._yscale_label(mode))
            ylim_btn = self._ylim_buttons.get(name)
            if ylim_btn is not None:
                ylim_mode = self._ylim_mode_by_var.get(name, "auto")
                ylim_btn.label.set_text(self._ylim_mode_label(ylim_mode))

            # Configure y-scale *before* plotting (more reliable than changing scale after plotting)
            # For symlog we pick linthresh from the (finite) data magnitude.
            linthresh = None
            try:
                if mode == "log":
                    ax.set_yscale("log")
                elif mode == "symlog":
                    # Estimate linthresh from data range at current time index across loaded cases
                    ys_all = []
                    for c in self.cases.values():
                        ds = c.ds
                        if name not in ds:
                            continue
                        da = ds[name]
                        # Clamp time index to case's available range
                        case_ti = self._get_time_index_for_case(c)
                        if tdim is not None and tdim in da.dims:
                            da1 = da.isel({tdim: case_ti})
                        else:
                            da1 = da
                        yv = np.asarray(da1.values)
                        yv = yv[np.isfinite(yv)]
                        if yv.size:
                            ys_all.append(yv)
                    if ys_all:
                        ys = np.concatenate(ys_all)
                        amax = float(np.nanmax(np.abs(ys))) if ys.size else 1.0
                        linthresh = max(1e-12, 1e-3 * amax)
                    else:
                        linthresh = 1e-6
                    ax.set_yscale("symlog", linthresh=linthresh)
                else:
                    ax.set_yscale("linear")
            except Exception as e:
                self.set_status(f"Y-scale error for {name}: {e}", is_error=True)

            # Extract units from first available dataset
            units = None
            for c in self.cases.values():
                ds = c.ds
                if name in ds:
                    try:
                        units = ds[name].attrs.get("units", None)
                        if units:
                            break
                    except Exception:
                        pass

            for c in self.cases.values():
                ds = c.ds
                if name not in ds:
                    continue
                da = ds[name]
                try:
                    # Clamp time index to case's available range
                    case_ti = self._get_time_index_for_case(c)
                    if tdim is not None and tdim in da.dims:
                        da1 = da.isel({tdim: case_ti})
                    else:
                        da1 = da
                    if sdim in ds.coords:
                        x = np.asarray(ds[sdim].values)
                    else:
                        x = np.arange(int(ds.sizes.get(sdim, da1.size)))
                    y = np.asarray(da1.values)
                    if mode == "log":
                        # Avoid matplotlib raising on non-positive values
                        y = np.where(y > 0, y, np.nan)
                    ax.plot(x, y, label=c.label)
                except Exception as e:
                    self.set_status(f"Plot error for {name}: {e}", is_error=True)

            # Set subplot title to variable name with padding
            ax.set_title(name, fontsize=10, pad=10)

            # Set y-label to units only (if available)
            ylabel = f"({units})" if units else ""
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.3)
            if len(self.cases) > 1:
                # Place legend in upper left to avoid y-scale button in upper right
                ax.legend(loc="upper left", fontsize=9)

            # Apply y-limit mode
            ylim_mode = self._ylim_mode_by_var.get(name, "auto")
            try:
                if ylim_mode == "auto":
                    # Dynamic: autoscale at each time step
                    ax.relim()
                    ax.autoscale_view()
                elif ylim_mode == "final":
                    # Fixed to final time step
                    ymin, ymax = self._compute_ylim_for_final(name, tdim, mode)
                    if ymin is not None and ymax is not None:
                        ax.set_ylim(ymin, ymax)
                    else:
                        ax.relim()
                        ax.autoscale_view()
                elif ylim_mode == "global":
                    # Fixed to global max/min
                    ymin, ymax = self._compute_ylim_for_global(name, mode)
                    if ymin is not None and ymax is not None:
                        ax.set_ylim(ymin, ymax)
                    else:
                        ax.relim()
                        ax.autoscale_view()

                # Add extra padding at the top to prevent button overlap with data
                # Calculate padding based on button height relative to axes height
                ymin, ymax = ax.get_ylim()
                pos = ax.get_position()  # axes position in figure coords
                button_height_fig = 0.032  # button height in figure coords
                axes_height_fig = pos.height
                # Button occupies this fraction of the axes
                button_frac = button_height_fig / axes_height_fig if axes_height_fig > 0 else 0.1
                # Add extra margin for safety
                padding_frac = button_frac + 0.05

                if mode == "log":
                    # For log scales, convert padding fraction to log space
                    if ymax > 0 and ymin > 0:
                        yrange_log = np.log10(ymax) - np.log10(ymin)
                        ymax_new = 10 ** (np.log10(ymax) + padding_frac * yrange_log)
                        ax.set_ylim(ymin, ymax_new)
                elif mode == "symlog":
                    # For symlog, add one order of magnitude at the top
                    try:
                        # If ymax is well into the logarithmic regime, add full decade
                        if abs(ymax) > 10:  # arbitrary threshold for log regime
                            if ymax > 0:
                                # Add one order of magnitude for button clearance
                                ymax_new = ymax * 10
                                ax.set_ylim(ymin, ymax_new)
                            else:
                                yrange = ymax - ymin
                                if yrange > 0:
                                    ax.set_ylim(ymin, ymax + padding_frac * yrange)
                        else:
                            # Linear regime - use linear padding
                            yrange = ymax - ymin
                            if yrange > 0:
                                ax.set_ylim(ymin, ymax + padding_frac * yrange)
                    except Exception:
                        # Fallback to linear padding
                        yrange = ymax - ymin
                        if yrange > 0:
                            ax.set_ylim(ymin, ymax + padding_frac * yrange)
                else:
                    # For linear scale, add padding based on button size
                    yrange = ymax - ymin
                    if yrange > 0:
                        ax.set_ylim(ymin, ymax + padding_frac * yrange)
            except Exception:
                pass

        # Put x-label on the bottom-most axis in each column
        n = len(vars_to_plot)
        nrows, ncols = getattr(self, "_last_plot_grid", (n, 1))
        nrows = max(1, int(nrows))
        ncols = max(1, int(ncols))
        bottom_axes: List[int] = []
        for col in range(ncols):
            inds = [i for i in range(n) if (i // nrows) == col]
            if inds:
                bottom_axes.append(max(inds))
        for i, ax in enumerate(self.axes):
            if i in bottom_axes:
                # ax.set_xlabel(sdim or "pos ")
                ax.set_xlabel(r'S$_\parallel$ (m)')
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes-3 1D GUI (Matplotlib widgets; no Tkinter).")
    parser.add_argument(
        "casepath",
        nargs="?",
        default=None,
        help="Path to Hermes-3 1D case directory (contains BOUT.dmp.*.nc and BOUT.inp).",
    )
    parser.add_argument("--n-plots", type=int, default=2, help="Number of subplots (default: 2).")
    parser.add_argument(
        "--spatial-dim",
        type=str,
        default=None,
        help="Force the spatial dimension name (default: infer, usually 'pos').",
    )
    args = parser.parse_args(argv)

    app = Hermes3GuiApp(
        initial_case_path=args.casepath,
        n_plots=args.n_plots,
        spatial_dim=args.spatial_dim,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



