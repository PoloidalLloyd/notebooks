"""
Hermes-3 1D GUI (PyQt + embedded Matplotlib).

This is a refactor of `hermes3_gui.py`, which implemented a "GUI" using
`matplotlib.widgets` (Buttons, Sliders, TextBox) inside a Matplotlib window.

This version uses Qt widgets for UI (path box, buttons, variable list, slider)
and embeds Matplotlib as a plotting canvas.

Run:

```bash
python hermes3_gui_pyqt.py /path/to/case_dir
```
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure we use a Qt backend for embedded Matplotlib.
import matplotlib

matplotlib.use("QtAgg", force=True)
from matplotlib import rcParams  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar  # noqa: E402


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


@dataclass
class _LoadedCase:
    label: str
    case_path: str
    ds: "object"  # xarray.Dataset (kept generic)
    n_time: int = 1


def _ensure_sdtools_on_path():
    """
    Make a best-effort attempt to ensure `analysis/sdtools` is importable.

    This script lives in: analysis/notebooks/hermes-3/general_functions/
    sdtools lives in:        analysis/sdtools/
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        sdtools_dir = parent / "analysis" / "sdtools"
        if sdtools_dir.exists():
            sp = str(sdtools_dir)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return
    for parent in [here.parent, *here.parents]:
        if parent.name == "analysis":
            sdtools_dir = parent / "sdtools"
            if sdtools_dir.exists():
                sp = str(sdtools_dir)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
            return


# ---- Qt imports (PyQt6 preferred; fall back to PySide6) ----
try:
    from PyQt6.QtCore import QEvent, Qt  # type: ignore
    from PyQt6.QtGui import QAction, QColor, QPalette  # type: ignore
    from PyQt6.QtWidgets import (  # type: ignore
        QAbstractItemView,
        QApplication,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QPushButton,
        QSlider,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    _QT_API = "PyQt6"

    def _qt_checked() -> "Qt.CheckState":
        return Qt.CheckState.Checked

    def _qt_unchecked() -> "Qt.CheckState":
        return Qt.CheckState.Unchecked

except Exception:  # pragma: no cover
    from PySide6.QtCore import QEvent, Qt  # type: ignore
    from PySide6.QtGui import QAction, QColor, QPalette  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QAbstractItemView,
        QApplication,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QPushButton,
        QSlider,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    _QT_API = "PySide6"

    def _qt_checked():
        return Qt.Checked

    def _qt_unchecked():
        return Qt.Unchecked


def _apply_mpl_light_theme() -> None:
    """
    Force Matplotlib to a light theme (white backgrounds / dark text).

    Matplotlib itself is not "aware" of the OS theme, but on macOS + Qt backends
    it can look dark if facecolors are not explicit.
    """
    rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222",
            "axes.labelcolor": "#111",
            "text.color": "#111",
            "xtick.color": "#111",
            "ytick.color": "#111",
            "grid.color": "#dddddd",
            "legend.facecolor": "white",
            "legend.edgecolor": "#cccccc",
        }
    )


def _apply_qt_light_theme(app: "QApplication") -> None:
    """
    Force the Qt application to a light palette (so the UI doesn't inherit macOS dark mode).
    """
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    p = QPalette()
    # Light palette (Qt docs style)
    p.setColor(QPalette.ColorRole.Window, QColor(245, 245, 245))
    p.setColor(QPalette.ColorRole.WindowText, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(240, 240, 240))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.Text, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.Button, QColor(245, 245, 245))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.BrightText, QColor(180, 0, 0))
    p.setColor(QPalette.ColorRole.Link, QColor(0, 90, 180))
    p.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    app.setPalette(p)


class Hermes3QtMainWindow(QMainWindow):
    """
    Qt GUI embedding Matplotlib for Hermes-3 1D profiles.

    Feature parity goals with the Matplotlib-widgets GUI:
    - Load/append datasets (multiple cases)
    - Searchable, checkable variable list; preserve selection order
    - Time slider (index), with readout (time coordinate if available)
    - Per-variable y-scale (linear/log/symlog) and y-limits mode (auto/final/global)

    Differences:
    - Per-variable scale/ylim controls are exposed via a right-click menu on the variable list
      and shown inline in the variable list item text (instead of Matplotlib overlay buttons).
    """

    def __init__(self, *, initial_case_path: Optional[str], spatial_dim: Optional[str]):
        super().__init__()

        _ensure_sdtools_on_path()
        try:
            from hermes3.load import Load  # type: ignore
        except Exception as e:
            raise ImportError(
                "Could not import `hermes3.load.Load`.\n"
                "Fix by either:\n"
                "- setting PYTHONPATH to include `.../analysis/sdtools`, or\n"
                "- running this script from within the repo where `analysis/sdtools` exists.\n"
                f"Original error: {e}"
            ) from e

        self.Load = Load

        self.setWindowTitle(f"Hermes-3 GUI (1D) - Qt ({_QT_API})")

        self.cases: Dict[str, _LoadedCase] = {}
        self.spatial_dim_forced = spatial_dim
        self.state = dict(spatial_dim=None, time_dim=None, vars=[], t_values=None)

        self.selected_vars: List[str] = []  # preserve selection order
        self._selected_set: set[str] = set()
        self._yscale_by_var: Dict[str, str] = {}  # var -> {"linear","log","symlog"}
        self._ylim_mode_by_var: Dict[str, str] = {}  # var -> {"auto","final","global"}
        self._var_filter: str = ""

        self._build_ui()

        # Overlay controls (Qt buttons positioned on top of each subplot).
        # var -> (ylim_button, yscale_button)
        self._overlay_buttons: Dict[str, Tuple["QPushButton", "QPushButton"]] = {}
        # var -> matplotlib Axes (for positioning)
        self._overlay_axes_by_var: Dict[str, "object"] = {}
        # Geometry constants (pixels, in canvas coordinates)
        self._overlay_btn_h = 22
        self._overlay_btn_w_yscale = 56
        self._overlay_btn_w_ylim = 72
        self._overlay_pad = 6

        # Keep overlay buttons positioned correctly on draw + resize.
        self.canvas.mpl_connect("draw_event", lambda _evt: self._position_overlay_buttons())
        self.canvas.installEventFilter(self)

        if initial_case_path:
            self.path_edit.setText(str(initial_case_path))
            self.load_dataset(replace=True)
        else:
            self.set_status("Enter a case directory path and click 'Load dataset'.")
            self.redraw()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        splitter = QSplitter()

        # Left panel
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        # Dataset path row
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("dataset path"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("/path/to/case_dir")
        path_row.addWidget(self.path_edit, 1)
        left_layout.addLayout(path_row)

        # Buttons row
        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load dataset")
        self.add_btn = QPushButton("Load additional")
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.add_btn)
        left_layout.addLayout(btn_row)

        # Status + datasets
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        self.datasets_label = QLabel("Loaded datasets: (none)")
        self.datasets_label.setWordWrap(True)
        left_layout.addWidget(self.datasets_label)

        # Search box
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("search variables…")
        left_layout.addWidget(self.search_edit)

        left_layout.addWidget(QLabel("Variables (check to plot; right-click for options)"))
        self.vars_list = QListWidget()
        self.vars_list.setUniformItemSizes(True)
        self.vars_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        # Make sure double-click doesn't try to edit labels
        try:
            self.vars_list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        except Exception:
            # PySide6 older enums fallback
            self.vars_list.setEditTriggers(QAbstractItemView.NoEditTriggers)  # type: ignore[attr-defined]
        left_layout.addWidget(self.vars_list, 1)

        self.deselect_btn = QPushButton("Deselect All")
        left_layout.addWidget(self.deselect_btn)

        # Right panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        self.figure = Figure(figsize=(10.5, 7.5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, 1)

        slider_row = QHBoxLayout()
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.setSingleStep(1)
        self.time_slider.setPageStep(1)
        self.time_slider.setValue(0)
        self.time_readout = QLabel("time index = 0")
        slider_row.addWidget(QLabel("time index"))
        slider_row.addWidget(self.time_slider, 1)
        slider_row.addWidget(self.time_readout)
        right_layout.addLayout(slider_row)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout = QVBoxLayout(root)
        main_layout.addWidget(splitter)

        # Wire signals
        self.load_btn.clicked.connect(lambda: self.load_dataset(replace=True))
        self.add_btn.clicked.connect(lambda: self.load_dataset(replace=False))
        self.deselect_btn.clicked.connect(self.deselect_all_vars)

        self.search_edit.textChanged.connect(self._on_search_change)
        self.vars_list.itemChanged.connect(self._on_var_item_changed)
        self.vars_list.itemDoubleClicked.connect(self._on_var_item_double_clicked)
        self.vars_list.customContextMenuRequested.connect(self._on_var_list_context_menu)
        self.time_slider.valueChanged.connect(lambda _v: self.redraw())

    # ---------- Status / datasets ----------
    def set_status(self, msg: str, *, is_error: bool = False) -> None:
        self.status_label.setText(msg)
        self.status_label.setStyleSheet("color: #b00020;" if is_error else "color: #333;")

    def _update_datasets_list(self) -> None:
        if not self.cases:
            self.datasets_label.setText("Loaded datasets: (none)")
            return
        labels = [c.label for c in self.cases.values()]
        if len(labels) <= 2:
            items = "\n".join(f"- {lbl}" for lbl in labels)
            self.datasets_label.setText(f"Loaded datasets ({len(labels)}):\n{items}")
        else:
            shown = "\n".join(f"- {lbl}" for lbl in labels[:2])
            self.datasets_label.setText(
                f"Loaded datasets ({len(labels)}):\n{shown}\n... and {len(labels) - 2} more"
            )

    # ---------- Variables list ----------
    def _on_search_change(self, text: str) -> None:
        self._var_filter = text or ""
        self._render_var_list()

    def _filtered_vars(self) -> List[str]:
        vars_all = list(self.state.get("vars") or [])
        q = (self._var_filter or "").strip().lower()
        if not q:
            return vars_all
        return [v for v in vars_all if q in v.lower()]

    def _item_text_for_var(self, name: str) -> str:
        ymode = self._yscale_by_var.get(name, "linear")
        ylim = self._ylim_mode_by_var.get(name, "auto")
        return f"{name}   [y:{ymode}, ylim:{ylim}]"

    def _render_var_list(self) -> None:
        self.vars_list.blockSignals(True)
        try:
            self.vars_list.clear()
            vars_all = list(self.state.get("vars") or [])
            if not vars_all:
                return
            for name in self._filtered_vars():
                item = QListWidgetItem(self._item_text_for_var(name))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(_qt_checked() if name in self._selected_set else _qt_unchecked())
                # Store the raw varname so label edits don't break lookups
                item.setData(Qt.ItemDataRole.UserRole, name)
                self.vars_list.addItem(item)
        finally:
            self.vars_list.blockSignals(False)

    def _find_item_by_var(self, varname: str) -> Optional["QListWidgetItem"]:
        for i in range(self.vars_list.count()):
            it = self.vars_list.item(i)
            if it is None:
                continue
            if it.data(Qt.ItemDataRole.UserRole) == varname:
                return it
        return None

    def _on_var_item_changed(self, item: "QListWidgetItem") -> None:
        name = item.data(Qt.ItemDataRole.UserRole)
        if not name:
            return
        checked = item.checkState() == _qt_checked()
        if checked and name not in self._selected_set:
            self._selected_set.add(name)
            self.selected_vars.append(name)
            self._yscale_by_var.setdefault(name, "linear")
            self._ylim_mode_by_var.setdefault(name, "auto")
        elif (not checked) and name in self._selected_set:
            self._selected_set.remove(name)
            self.selected_vars = [v for v in self.selected_vars if v != name]

        # Update display text (so mode info stays visible)
        item.setText(self._item_text_for_var(name))
        self.redraw()

    def _on_var_item_double_clicked(self, item: "QListWidgetItem") -> None:
        """
        Double-clicking anywhere on the row (including the text) toggles the checkbox.
        """
        try:
            cur = item.checkState()
            nxt = _qt_unchecked() if cur == _qt_checked() else _qt_checked()
            item.setCheckState(nxt)  # triggers _on_var_item_changed
        except Exception:
            pass

    def deselect_all_vars(self) -> None:
        self._selected_set = set()
        self.selected_vars = []
        self.vars_list.blockSignals(True)
        try:
            for i in range(self.vars_list.count()):
                it = self.vars_list.item(i)
                if it is not None:
                    it.setCheckState(_qt_unchecked())
        finally:
            self.vars_list.blockSignals(False)
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

    def _on_var_list_context_menu(self, pos) -> None:
        item = self.vars_list.itemAt(pos)
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        if not name:
            return

        menu = QMenu(self)

        act_cycle_y = QAction("Cycle y-scale (linear → log → symlog)", self)
        act_cycle_ylim = QAction("Cycle y-limits (auto → final → global)", self)
        menu.addAction(act_cycle_y)
        menu.addAction(act_cycle_ylim)
        menu.addSeparator()

        # Explicit set menus
        m_y = menu.addMenu("Set y-scale")
        for mode in ("linear", "log", "symlog"):
            a = QAction(mode, self)
            a.setCheckable(True)
            a.setChecked(self._yscale_by_var.get(name, "linear") == mode)
            a.triggered.connect(lambda _=False, m=mode: self._set_var_yscale(name, m))
            m_y.addAction(a)

        m_ylim = menu.addMenu("Set y-limits")
        for mode in ("auto", "final", "global"):
            a = QAction(mode, self)
            a.setCheckable(True)
            a.setChecked(self._ylim_mode_by_var.get(name, "auto") == mode)
            a.triggered.connect(lambda _=False, m=mode: self._set_var_ylim_mode(name, m))
            m_ylim.addAction(a)

        def _do_cycle_y():
            cur = self._yscale_by_var.get(name, "linear")
            self._yscale_by_var[name] = self._cycle_yscale(cur)
            self._refresh_var_item(name)
            self.redraw()

        def _do_cycle_ylim():
            cur = self._ylim_mode_by_var.get(name, "auto")
            self._ylim_mode_by_var[name] = self._cycle_ylim_mode(cur)
            self._refresh_var_item(name)
            self.redraw()

        act_cycle_y.triggered.connect(_do_cycle_y)
        act_cycle_ylim.triggered.connect(_do_cycle_ylim)

        menu.exec(self.vars_list.mapToGlobal(pos))

    def _refresh_var_item(self, varname: str) -> None:
        it = self._find_item_by_var(varname)
        if it is None:
            return
        self.vars_list.blockSignals(True)
        try:
            it.setText(self._item_text_for_var(varname))
        finally:
            self.vars_list.blockSignals(False)

    def _set_var_yscale(self, varname: str, mode: str) -> None:
        self._yscale_by_var[varname] = mode
        self._refresh_var_item(varname)
        self._refresh_overlay_button_labels(varname)
        self.redraw()

    def _set_var_ylim_mode(self, varname: str, mode: str) -> None:
        self._ylim_mode_by_var[varname] = mode
        self._refresh_var_item(varname)
        self._refresh_overlay_button_labels(varname)
        self.redraw()

    # ---------- Overlay buttons on plots (Option B) ----------
    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        """
        Reposition overlay buttons when the canvas resizes.
        """
        try:
            if obj is self.canvas:
                et = event.type()
                # PyQt6/PySide6 both expose QEvent.Type.Resize; keep a fallback just in case.
                if et == QEvent.Type.Resize or et == getattr(QEvent, "Resize", None):
                    self._position_overlay_buttons()
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _clear_overlay_buttons(self) -> None:
        for ylim_btn, yscale_btn in list(self._overlay_buttons.values()):
            try:
                ylim_btn.hide()
                yscale_btn.hide()
                ylim_btn.deleteLater()
                yscale_btn.deleteLater()
            except Exception:
                pass
        self._overlay_buttons = {}
        self._overlay_axes_by_var = {}

    def _sync_overlay_buttons(self, vars_to_plot: List[str], axes: List["object"]) -> None:
        """
        Ensure we have one pair of overlay buttons per variable being plotted,
        and keep a mapping from var -> axes for positioning.
        """
        # Remove buttons for vars no longer plotted
        keep = set(vars_to_plot)
        for v in list(self._overlay_buttons.keys()):
            if v not in keep:
                try:
                    ylim_btn, yscale_btn = self._overlay_buttons.pop(v)
                    ylim_btn.hide()
                    yscale_btn.hide()
                    ylim_btn.deleteLater()
                    yscale_btn.deleteLater()
                except Exception:
                    pass
                self._overlay_axes_by_var.pop(v, None)

        # Update axes mapping (zip in selection order)
        self._overlay_axes_by_var = {v: ax for v, ax in zip(vars_to_plot, axes)}

        # Create buttons for any new vars
        for v in vars_to_plot:
            if v in self._overlay_buttons:
                self._refresh_overlay_button_labels(v)
                continue

            # Create as children of the canvas so they overlay the plot area.
            ylim_btn = QPushButton(self.canvas)
            yscale_btn = QPushButton(self.canvas)

            ylim_btn.setText(self._ylim_mode_label(self._ylim_mode_by_var.get(v, "auto")))
            yscale_btn.setText(self._yscale_label(self._yscale_by_var.get(v, "linear")))

            # Make small and unobtrusive
            ylim_btn.setFixedHeight(self._overlay_btn_h)
            yscale_btn.setFixedHeight(self._overlay_btn_h)
            ylim_btn.setFixedWidth(self._overlay_btn_w_ylim)
            yscale_btn.setFixedWidth(self._overlay_btn_w_yscale)

            # Slightly transparent background so data remains visible.
            # (Qt style sheets are safe; if ignored, it's fine.)
            try:
                style = (
                    "QPushButton {"
                    " background: rgba(250, 250, 250, 210);"
                    " border: 1px solid rgba(0,0,0,80);"
                    " border-radius: 4px;"
                    " padding: 1px 4px;"
                    " font-size: 10px;"
                    "}"
                    "QPushButton:pressed { background: rgba(230, 230, 230, 230); }"
                )
                ylim_btn.setStyleSheet(style)
                yscale_btn.setStyleSheet(style)
            except Exception:
                pass

            # Click actions
            ylim_btn.clicked.connect(partial(self._on_overlay_ylim_clicked, v))
            yscale_btn.clicked.connect(partial(self._on_overlay_yscale_clicked, v))

            ylim_btn.show()
            yscale_btn.show()
            ylim_btn.raise_()
            yscale_btn.raise_()

            self._overlay_buttons[v] = (ylim_btn, yscale_btn)

        # Position now (and again on draw_event/resize).
        self._position_overlay_buttons()

    def _refresh_overlay_button_labels(self, varname: str) -> None:
        pair = self._overlay_buttons.get(varname)
        if not pair:
            return
        ylim_btn, yscale_btn = pair
        ylim_btn.setText(self._ylim_mode_label(self._ylim_mode_by_var.get(varname, "auto")))
        yscale_btn.setText(self._yscale_label(self._yscale_by_var.get(varname, "linear")))

    def _on_overlay_yscale_clicked(self, varname: str) -> None:
        cur = self._yscale_by_var.get(varname, "linear")
        self._yscale_by_var[varname] = self._cycle_yscale(cur)
        self._refresh_var_item(varname)
        self._refresh_overlay_button_labels(varname)
        self.redraw()

    def _on_overlay_ylim_clicked(self, varname: str) -> None:
        cur = self._ylim_mode_by_var.get(varname, "auto")
        self._ylim_mode_by_var[varname] = self._cycle_ylim_mode(cur)
        self._refresh_var_item(varname)
        self._refresh_overlay_button_labels(varname)
        self.redraw()

    def _position_overlay_buttons(self) -> None:
        """
        Position overlay buttons in canvas pixel coordinates.

        Matplotlib Axes positions are in figure fraction coordinates with origin at bottom-left.
        Qt widget positions are in pixels with origin at top-left.
        """
        if not self._overlay_buttons or not self._overlay_axes_by_var:
            return

        try:
            w, h = self.canvas.get_width_height()
        except Exception:
            return
        if not w or not h:
            return

        pad = int(self._overlay_pad)
        bh = int(self._overlay_btn_h)
        bw_y = int(self._overlay_btn_w_yscale)
        bw_l = int(self._overlay_btn_w_ylim)

        for v, (ylim_btn, yscale_btn) in list(self._overlay_buttons.items()):
            ax = self._overlay_axes_by_var.get(v)
            if ax is None:
                try:
                    ylim_btn.hide()
                    yscale_btn.hide()
                except Exception:
                    pass
                continue

            try:
                pos = ax.get_position()  # figure fraction coords
                x_right = int(pos.x1 * w)
                y_top = int((1.0 - pos.y1) * h)
            except Exception:
                continue

            # Place yscale at top-right inside axes; ylim just to its left.
            y = max(0, y_top + pad)
            x_yscale = max(0, x_right - bw_y - pad)
            x_ylim = max(0, x_yscale - bw_l - pad)

            try:
                yscale_btn.setGeometry(x_yscale, y, bw_y, bh)
                ylim_btn.setGeometry(x_ylim, y, bw_l, bh)
                ylim_btn.show()
                yscale_btn.show()
                ylim_btn.raise_()
                yscale_btn.raise_()
            except Exception:
                pass

    # ---------- Data loading ----------
    def _load_case(self, case_path: str) -> _LoadedCase:
        case_path = str(Path(case_path).expanduser().resolve())
        label = _format_case_label(case_path)
        cs = self.Load.case_1D(case_path, verbose=False)
        tdim = _infer_time_dim(cs.ds)
        n_time = int(cs.ds.sizes[tdim]) if tdim and tdim in cs.ds.dims else 1
        return _LoadedCase(label=label, case_path=case_path, ds=cs.ds, n_time=n_time)

    def _recompute_all_vars(self) -> Tuple[List[str], Optional[str], Optional[str]]:
        if not self.cases:
            return [], None, None
        first = next(iter(self.cases.values())).ds
        tdim = _infer_time_dim(first)
        sdim = self.spatial_dim_forced or _infer_spatial_dim(first)

        all_vars = set(_list_plottable_vars(first, spatial_dim=sdim, time_dim=tdim))
        for c in list(self.cases.values())[1:]:
            all_vars |= set(_list_plottable_vars(c.ds, spatial_dim=sdim, time_dim=tdim))
        return sorted(all_vars), sdim, tdim

    def _set_time_range(self, n_t: int) -> None:
        n_t = max(1, int(n_t))
        self.time_slider.blockSignals(True)
        try:
            self.time_slider.setMinimum(0)
            self.time_slider.setMaximum(max(0, n_t - 1))
            # set to final time step by default
            self.time_slider.setValue(n_t - 1)
        finally:
            self.time_slider.blockSignals(False)

    def _update_after_load(self) -> None:
        vars_, sdim, tdim = self._recompute_all_vars()
        self.state["vars"] = vars_
        self.state["spatial_dim"] = sdim
        self.state["time_dim"] = tdim

        # Drop selections that no longer exist
        if vars_:
            keep = [v for v in self.selected_vars if v in vars_]
            self.selected_vars = keep
            self._selected_set = set(keep)
            if not self.selected_vars:
                default_var = "Te" if "Te" in vars_ else vars_[0]
                self.selected_vars = [default_var]
                self._selected_set = {default_var}
                self._yscale_by_var.setdefault(default_var, "linear")
                self._ylim_mode_by_var.setdefault(default_var, "auto")
        else:
            self.selected_vars = []
            self._selected_set = set()

        # Time axis values from the first dataset (for display)
        ds0 = next(iter(self.cases.values())).ds
        t_values = None
        if tdim is not None and tdim in ds0.coords:
            try:
                t_values = np.asarray(ds0[tdim].values)
            except Exception:
                t_values = None
        self.state["t_values"] = t_values

        # Slider range based on maximum time steps across cases
        max_n_t = max((c.n_time for c in self.cases.values()), default=1)
        self._set_time_range(max_n_t)

        self._render_var_list()

    def load_dataset(self, *, replace: bool) -> None:
        p = (self.path_edit.text() or "").strip()
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
            self.set_status("")
            self.redraw()
        except Exception as e:
            self.set_status(f"Failed to load dataset: {e}", is_error=True)

    # ---------- Plotting ----------
    def _get_time_index(self) -> int:
        try:
            return int(self.time_slider.value())
        except Exception:
            return 0

    def _get_time_index_for_case(self, case: _LoadedCase) -> int:
        ti = self._get_time_index()
        return min(ti, case.n_time - 1)

    def _update_time_readout(self) -> None:
        ti = self._get_time_index()
        tdim = self.state.get("time_dim")
        tvals = self.state.get("t_values")
        if tdim and tvals is not None and ti < len(tvals):
            try:
                self.time_readout.setText(f"{tdim} = {tvals[ti] * 1e3:.4f} ms")
                return
            except Exception:
                pass
        self.time_readout.setText(f"time index = {ti}")

    def _compute_ylim_for_final(self, varname: str, tdim: Optional[str], yscale: str) -> Tuple[Optional[float], Optional[float]]:
        ys_all = []
        for c in self.cases.values():
            ds = c.ds
            if varname not in ds:
                continue
            da = ds[varname]
            try:
                final_ti = c.n_time - 1
                if tdim is not None and tdim in da.dims:
                    da1 = da.isel({tdim: final_ti})
                else:
                    da1 = da
                yv = np.asarray(da1.values)
                yv = yv[np.isfinite(yv)]
                if yscale == "log":
                    yv = yv[yv > 0]
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
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.1 * abs(ymax)
        return ymin - margin, ymax + margin

    def _compute_ylim_for_global(self, varname: str, yscale: str) -> Tuple[Optional[float], Optional[float]]:
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
                    yv = yv[yv > 0]
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
        margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.1 * abs(ymax)
        return ymin - margin, ymax + margin

    def redraw(self) -> None:
        self._update_time_readout()

        self.figure.clear()
        # Explicit facecolor to avoid inheriting dark appearances on some platforms.
        try:
            self.figure.set_facecolor("white")
        except Exception:
            pass

        if not self.cases:
            # No plots -> no overlay buttons
            self._clear_overlay_buttons()
            ax = self.figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No dataset loaded.\nLoad a case directory to view variables.",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            self.canvas.draw_idle()
            return

        sdim = self.state.get("spatial_dim")
        tdim = self.state.get("time_dim")
        vars_to_plot = list(self.selected_vars)

        if not vars_to_plot:
            self._clear_overlay_buttons()
            ax = self.figure.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No variables selected.\nCheck variables on the left to plot.",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            self.canvas.draw_idle()
            return

        n = len(vars_to_plot)
        nrows = min(3, n)
        ncols = int(np.ceil(n / nrows))

        gs = self.figure.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.35, wspace=0.30)

        sharex_ref: List[Optional["object"]] = [None] * ncols
        axes: List["object"] = []

        for idx in range(n):
            col = idx // nrows
            row = idx % nrows
            sharex = sharex_ref[col]
            ax = self.figure.add_subplot(gs[row, col], sharex=sharex)
            if sharex_ref[col] is None:
                sharex_ref[col] = ax
            axes.append(ax)

        # Determine bottom-most axis per column for x-label and tick labels
        bottom_idx_by_col: Dict[int, int] = {}
        for col in range(ncols):
            inds = [i for i in range(n) if (i // nrows) == col]
            if inds:
                bottom_idx_by_col[col] = max(inds)

        for i, ax in enumerate(axes):
            col = i // nrows
            is_bottom = bottom_idx_by_col.get(col, -1) == i
            if not is_bottom:
                ax.tick_params(labelbottom=False)

        for ax, name in zip(axes, vars_to_plot):
            mode = self._yscale_by_var.get(name, "linear")
            ylim_mode = self._ylim_mode_by_var.get(name, "auto")

            # Configure y-scale before plotting
            linthresh = None
            try:
                if mode == "log":
                    ax.set_yscale("log")
                elif mode == "symlog":
                    ys_all = []
                    for c in self.cases.values():
                        ds = c.ds
                        if name not in ds:
                            continue
                        da = ds[name]
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

            # Extract units from first dataset that has var
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
                    case_ti = self._get_time_index_for_case(c)
                    if tdim is not None and tdim in da.dims:
                        da1 = da.isel({tdim: case_ti})
                    else:
                        da1 = da

                    if sdim and sdim in ds.coords:
                        x = np.asarray(ds[sdim].values)
                    else:
                        x = np.arange(int(ds.sizes.get(sdim, da1.size))) if sdim else np.arange(da1.size)

                    y = np.asarray(da1.values)
                    if mode == "log":
                        y = np.where(y > 0, y, np.nan)
                    ax.plot(x, y, label=c.label)
                except Exception as e:
                    self.set_status(f"Plot error for {name}: {e}", is_error=True)

            ax.set_title(name, fontsize=10)
            ax.set_ylabel(f"({units})" if units else "")
            ax.grid(True, which="both", alpha=0.3)
            if len(self.cases) > 1:
                ax.legend(loc="upper left", fontsize=9)

            # Apply y-limit mode
            try:
                if ylim_mode == "auto":
                    ax.relim()
                    ax.autoscale_view()
                elif ylim_mode == "final":
                    ymin, ymax = self._compute_ylim_for_final(name, tdim, mode)
                    if ymin is not None and ymax is not None:
                        ax.set_ylim(ymin, ymax)
                    else:
                        ax.relim()
                        ax.autoscale_view()
                elif ylim_mode == "global":
                    ymin, ymax = self._compute_ylim_for_global(name, mode)
                    if ymin is not None and ymax is not None:
                        ax.set_ylim(ymin, ymax)
                    else:
                        ax.relim()
                        ax.autoscale_view()
            except Exception:
                pass

        # X label on bottom-most axis in each column
        for i, ax in enumerate(axes):
            col = i // nrows
            is_bottom = bottom_idx_by_col.get(col, -1) == i
            if is_bottom:
                ax.set_xlabel(r"S$_\parallel$ (m)")

        # Sync overlay buttons to the current subplot grid.
        self._sync_overlay_buttons(vars_to_plot=vars_to_plot, axes=axes)
        self.canvas.draw_idle()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes-3 1D GUI (PyQt + embedded Matplotlib).")
    parser.add_argument(
        "casepath",
        nargs="?",
        default=None,
        help="Path to Hermes-3 1D case directory (contains BOUT.dmp.*.nc and BOUT.inp).",
    )
    parser.add_argument(
        "--spatial-dim",
        type=str,
        default=None,
        help="Force the spatial dimension name (default: infer, usually 'pos').",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="light",
        choices=("system", "light", "dark"),
        help="GUI theme override. 'system' follows OS, 'light' forces light mode.",
    )
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv)
    if args.theme == "light":
        _apply_qt_light_theme(app)
        _apply_mpl_light_theme()
    elif args.theme == "dark":
        # Keep Qt system palette by default; for Matplotlib, a dark theme is available.
        # If you want Qt dark, we'd set a dark QPalette here.
        rcParams.update(
            {
                "figure.facecolor": "#111",
                "savefig.facecolor": "#111",
                "axes.facecolor": "#111",
                "axes.edgecolor": "#ddd",
                "axes.labelcolor": "#eee",
                "text.color": "#eee",
                "xtick.color": "#eee",
                "ytick.color": "#eee",
                "grid.color": "#444",
                "legend.facecolor": "#111",
                "legend.edgecolor": "#444",
            }
        )
    win = Hermes3QtMainWindow(initial_case_path=args.casepath, spatial_dim=args.spatial_dim)
    win.resize(1400, 850)
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())

