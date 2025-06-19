# main_window.py - Versión final

import sys
# ⚠️ Esto soluciona problemas de icono/doble icono en Mac (añádelo justo aquí)
if sys.platform == "darwin":
    import os
    os.environ["QT_MAC_WANTS_LAYER"] = "1"
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QSplitter,
    QSizePolicy, QComboBox, QScrollArea, QMessageBox,
    QProgressDialog, QSlider
)

from image_processing import enhance_rock_art


class NoWheelSlider(QSlider):
    """Slider horizontal que ignora eventos de rueda."""
    def wheelEvent(self, event):
        event.ignore()


class ImageUtils:
    @staticmethod
    def np_to_qpixmap(np_img: np.ndarray) -> QPixmap:
        """Convierte ndarray RGB uint8 a QPixmap."""
        h, w, ch = np_img.shape
        bytes_per_line = w * ch
        q_img = QImage(np_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)


class ControlGroup:
    def __init__(self, title: str):
        self.group_box = QGroupBox(title)
        font = self.group_box.font()
        font.setBold(True)
        self.group_box.setFont(font)
        self.group_box.setAlignment(Qt.AlignCenter)
        self.layout = QGridLayout()
        self.layout.setVerticalSpacing(8)
        self.group_box.setLayout(self.layout)

    def add_slider(self, label: str, minimum: int, maximum: int,
                   default: int, tooltip: str, callback, row: int) -> QSlider:
        # Texto descriptivo
        lbl = QLabel(label)
        font_lbl = lbl.font()
        font_lbl.setBold(True)
        lbl.setFont(font_lbl)
        self.layout.addWidget(lbl, row, 0, alignment=Qt.AlignVCenter)

        # Slider
        slider = NoWheelSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(default)
        slider.setToolTip(tooltip)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.NoTicks)
        slider.setSingleStep(1)
        slider.setStyleSheet(
            "QSlider::handle:horizontal { width: 6px; height: 6px; margin: -3px 0; }"
        )
        self.layout.addWidget(slider, row, 1, alignment=Qt.AlignVCenter)

        # Etiqueta de valor al final del slider
        value_label = QLabel(str(default))
        font_val = value_label.font()
        font_val.setBold(False)
        font_val.setPointSize(max(8, font_val.pointSize() - 2))
        value_label.setFont(font_val)
        value_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(value_label, row, 2, alignment=Qt.AlignVCenter)

        # Actualizar valor y ejecutar callback
        def _on_change(val):
            value_label.setText(str(val))
            callback()
        slider.valueChanged.connect(_on_change)

        return slider

    def add_combo(self, label: str, items: list, default_index: int,
                  tooltip: str, callback, row: int) -> QComboBox:
        lbl = QLabel(label)
        font_lbl = lbl.font()
        font_lbl.setBold(True)
        lbl.setFont(font_lbl)
        self.layout.addWidget(lbl, row, 0, alignment=Qt.AlignVCenter)

        combo = QComboBox()
        combo.addItems(items)
        combo.setCurrentIndex(default_index)
        combo.setToolTip(tooltip)
        combo.currentIndexChanged.connect(lambda idx: callback())
        self.layout.addWidget(combo, row, 1, alignment=Qt.AlignVCenter)

        spacer = QWidget()
        spacer.setFixedWidth(40)
        self.layout.addWidget(spacer, row, 2)
        return combo

    def add_reset_button(self, text: str, tooltip: str, callback, row: int) -> QPushButton:
        btn = QPushButton(text)
        font_btn = btn.font()
        font_btn.setBold(False)
        btn.setFont(font_btn)
        btn.setToolTip(tooltip)
        btn.clicked.connect(callback)
        self.layout.addWidget(btn, row, 0, 1, 3, alignment=Qt.AlignCenter)
        return btn


class SolStretchProApp(QMainWindow):
    WINDOW_TITLE = "SolStretchPro - Rock Art Enhancer"
    WINDOW_SIZE = (1200, 800)
    SUPPORTED_FORMATS = "Images (*.jpg *.jpeg *.tif *.tiff)"
    DEFAULTS = {
        'blur_sigma': 25,
        'low_pct': 1,
        'high_pct': 99,
        'clahe_clip': 20,
        'clahe_grid': (8, 8),
        'gamma': 100,
        'brightness': 0,
        'contrast': 100,
        'r_gain': 100,
        'g_gain': 100,
        'b_gain': 100
    }

    def __init__(self):
        super().__init__()
        self.orig_img: Optional[np.ndarray] = None
        self.proc_img: Optional[np.ndarray] = None
        self._setup_ui()
        self._reset_all()

    def _setup_ui(self) -> None:
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(*self.WINDOW_SIZE)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.addLayout(self._create_header())
        main_layout.addWidget(self._create_content(), stretch=1)
        main_layout.addLayout(self._create_footer())

    def _reset_all(self) -> None:
        self.sld_blur.setValue(self.DEFAULTS['blur_sigma'])
        self.sld_lowpct.setValue(self.DEFAULTS['low_pct'])
        self.sld_highpct.setValue(self.DEFAULTS['high_pct'])
        self.sld_clahe.setValue(self.DEFAULTS['clahe_clip'])
        idx = ["4×4", "8×8", "16×16"].index(
            f"{self.DEFAULTS['clahe_grid'][0]}×{self.DEFAULTS['clahe_grid'][1]}"
        )
        self.cmb_clahe.setCurrentIndex(idx)
        self.sld_gamma.setValue(self.DEFAULTS['gamma'])
        self.sld_bright.setValue(self.DEFAULTS['brightness'])
        self.sld_contrast.setValue(self.DEFAULTS['contrast'])
        self.sld_rg.setValue(self.DEFAULTS['r_gain'])
        self.sld_gg.setValue(self.DEFAULTS['g_gain'])
        self.sld_bg.setValue(self.DEFAULTS['b_gain'])

    def _create_header(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # Imagen de encabezado: Rinoceronte_Libia_stretched.jpg
        header_img = Path(__file__).parent / "Rinoceronte_Libia_stretched.jpg"
        if header_img.exists():
            pm = QPixmap(str(header_img)).scaledToHeight(80, Qt.SmoothTransformation)
            lbl = QLabel()
            lbl.setPixmap(pm)
            lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            layout.addWidget(lbl)
        # Título de la aplicación
        title = QLabel(self.WINDOW_TITLE)
        font = QFont("Arial", 16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        layout.addStretch()
        return layout

    def _create_content(self) -> QSplitter:
        splitter = QSplitter(Qt.Horizontal)
        controls = self._create_controls()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        splitter.addWidget(scroll)
        splitter.addWidget(self._create_preview())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        return splitter

    def _create_controls(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        btn_open = QPushButton("Open Image")
        btn_open.setMaximumWidth(120)
        f = btn_open.font()
        f.setBold(False)
        btn_open.setFont(f)
        btn_open.clicked.connect(self.open_image)
        layout.addWidget(btn_open)

        # Background
        cg = ControlGroup("Background Subtraction")
        self.sld_blur = cg.add_slider("Blur Sigma", 0, 50, self.DEFAULTS['blur_sigma'],
            "Blur sigma", self.process_image, 0)
        cg.add_reset_button("Reset Background", "Reset sigma",
            lambda: self._reset('blur_sigma'), 1)
        layout.addWidget(cg.group_box)

        # PCA
        cg = ControlGroup("PCA Stretch (CIELAB)")
        self.sld_lowpct = cg.add_slider("Low Percentile (%)", 0, 20,
            self.DEFAULTS['low_pct'], "Low PCA percentile", self.process_image, 0)
        self.sld_highpct = cg.add_slider("High Percentile (%)", 80, 100,
            self.DEFAULTS['high_pct'], "High PCA percentile", self.process_image, 1)
        cg.add_reset_button("Reset Percentiles", "Reset PCA",
            lambda: self._reset('percentiles'), 2)
        layout.addWidget(cg.group_box)

        # CLAHE
        cg = ControlGroup("CLAHE Settings")
        self.sld_clahe = cg.add_slider("CLAHE Clip ×0.1", 10, 50,
            self.DEFAULTS['clahe_clip'], "CLAHE clip", self.process_image, 0)
        self.cmb_clahe = cg.add_combo("Tile Grid Size", ["4×4","8×8","16×16"], 1,
            "CLAHE grid size", self.process_image, 1)
        cg.add_reset_button("Reset CLAHE", "Reset CLAHE",
            lambda: self._reset('clahe'), 2)
        layout.addWidget(cg.group_box)

        # Gamma Correction
        cg = ControlGroup("Gamma Correction")
        self.sld_gamma = cg.add_slider("Gamma ×0.01", 50, 200,
            self.DEFAULTS['gamma'], "Gamma factor", self.process_image, 0)
        cg.add_reset_button("Reset Gamma", "Reset gamma",
            lambda: self._reset('gamma'), 1)
        layout.addWidget(cg.group_box)

        # Brightness / Contrast
        cg = ControlGroup("Brightness / Contrast")
        self.sld_bright = cg.add_slider("Brightness", -100, 100,
            self.DEFAULTS['brightness'], "Brightness adjustment", self.process_image, 0)
        self.sld_contrast = cg.add_slider("Contrast ×0.01", 10, 300,
            self.DEFAULTS['contrast'], "Contrast adjustment", self.process_image, 1)
        cg.add_reset_button("Reset B/C", "Reset Brightness & Contrast",
            lambda: self._reset('bc'), 2)
        layout.addWidget(cg.group_box)

        # RGB Channel Gains
        cg = ControlGroup("RGB Channel Gains")
        self.sld_rg = cg.add_slider("Red ×0.01", 0, 200,
            self.DEFAULTS['r_gain'], "Red gain", self.process_image, 0)
        self.sld_gg = cg.add_slider("Green ×0.01", 0, 200,
            self.DEFAULTS['g_gain'], "Green gain", self.process_image, 1)
        self.sld_bg = cg.add_slider("Blue ×0.01", 0, 200,
            self.DEFAULTS['b_gain'], "Blue gain", self.process_image, 2)
        cg.add_reset_button("Reset RGB", "Reset RGB gains",
            lambda: self._reset('rgb'), 3)
        layout.addWidget(cg.group_box)

        # Save button
        btn_save = QPushButton("Save Image")
        btn_save.setMaximumWidth(120)
        fs = btn_save.font()
        fs.setBold(False)
        btn_save.setFont(fs)
        btn_save.clicked.connect(self.save_image)
        layout.addWidget(btn_save)
        layout.addStretch()
        return w

    def _create_preview(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Original Image"))
        self.lbl_orig = QLabel()
        self.lbl_orig.setAlignment(Qt.AlignCenter)
        self.lbl_orig.setStyleSheet("border:1px solid gray;")
        layout.addWidget(self.lbl_orig, stretch=1)
        layout.addWidget(QLabel("Processed Image"))
        self.lbl_proc = QLabel()
        self.lbl_proc.setAlignment(Qt.AlignCenter)
        self.lbl_proc.setStyleSheet("border:1px solid gray;")
        layout.addWidget(self.lbl_proc, stretch=1)
        return w

    def _create_footer(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addStretch()
        pic = Path(__file__).parent / "SolStretch_logo.png"
        if pic.exists():
            pm = QPixmap(str(pic)).scaledToHeight(30, Qt.SmoothTransformation)
            lb = QLabel()
            lb.setPixmap(pm)
            layout.addWidget(lb)
        txt = QLabel("Powered by Solanako")
        font_txt = QFont("Arial", 8)
        font_txt.setItalic(True)
        txt.setFont(font_txt)
        layout.addWidget(txt)
        layout.addStretch()
        return layout

    def _reset(self, key: str) -> None:
        if key == 'blur_sigma':
            self.sld_blur.setValue(self.DEFAULTS['blur_sigma'])
        elif key == 'percentiles':
            self.sld_lowpct.setValue(self.DEFAULTS['low_pct'])
            self.sld_highpct.setValue(self.DEFAULTS['high_pct'])
        elif key == 'clahe':
            self.sld_clahe.setValue(self.DEFAULTS['clahe_clip'])
            idx = ["4×4","8×8","16×16"].index(
                f"{self.DEFAULTS['clahe_grid'][0]}×{self.DEFAULTS['clahe_grid'][1]}"
            )
            self.cmb_clahe.setCurrentIndex(idx)
        elif key == 'gamma':
            self.sld_gamma.setValue(self.DEFAULTS['gamma'])
        elif key == 'bc':
            self.sld_bright.setValue(self.DEFAULTS['brightness'])
            self.sld_contrast.setValue(self.DEFAULTS['contrast'])
        elif key == 'rgb':
            self.sld_rg.setValue(self.DEFAULTS['r_gain'])
            self.sld_gg.setValue(self.DEFAULTS['g_gain'])
            self.sld_bg.setValue(self.DEFAULTS['b_gain'])
        self.process_image()

    def open_image(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Open Image", "", self.SUPPORTED_FORMATS)
        if not fn:
            return
        img = Image.open(fn).convert("RGB")
        self.orig_img = np.array(img)
        pm = ImageUtils.np_to_qpixmap(self.orig_img)
        self.lbl_orig.setPixmap(pm.scaled(self.lbl_orig.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.process_image()

    def process_image(self) -> None:
        if self.orig_img is None:
            return
        params = self._get_params()
        dlg = QProgressDialog("Processing image...", None, 0, 0, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.show()
        QApplication.processEvents()
        out = enhance_rock_art(
            self.orig_img,
            blur_sigma=params['blur_sigma'],
            low_pct=params['low_pct'],
            high_pct=params['high_pct'],
            clahe_clip=params['clahe_clip'],
            clahe_grid=params['clahe_grid'],
            gamma=params['gamma']
        )
        dlg.close()
        out = self._apply_bc(out, params)
        out = self._apply_rgb(out, params)
        self.proc_img = out
        pm = ImageUtils.np_to_qpixmap(out)
        self.lbl_proc.setPixmap(pm.scaled(self.lbl_proc.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _get_params(self) -> Dict[str, Any]:
        grid = tuple(map(int, self.cmb_clahe.currentText().split('×')))
        return {
            'blur_sigma': float(self.sld_blur.value()),
            'low_pct': float(self.sld_lowpct.value()),
            'high_pct': float(self.sld_highpct.value()),
            'clahe_clip': float(self.sld_clahe.value())/10.0,
            'clahe_grid': grid,
            'gamma': float(self.sld_gamma.value())/100.0,
            'brightness': float(self.sld_bright.value()),
            'contrast': float(self.sld_contrast.value())/100.0,
            'r_gain': float(self.sld_rg.value())/100.0,
            'g_gain': float(self.sld_gg.value())/100.0,
            'b_gain': float(self.sld_bg.value())/100.0
        }

    def _apply_bc(self, img: np.ndarray, prm: Dict[str, Any]) -> np.ndarray:
        result = img.astype(np.float32)
        if prm['brightness'] != 0:
            result += (prm['brightness'] / 100.0) * 255
        if prm['contrast'] != 1:
            result = (result - 127.5) * prm['contrast'] + 127.5
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_rgb(self, img: np.ndarray, prm: Dict[str, Any]) -> np.ndarray:
        result = img.astype(np.float32)
        result[..., 0] = np.clip(result[..., 0] * prm['r_gain'], 0, 255)
        result[..., 1] = np.clip(result[..., 1] * prm['g_gain'], 0, 255)
        result[..., 2] = np.clip(result[..., 2] * prm['b_gain'], 0, 255)
        return result.astype(np.uint8)

    def save_image(self) -> None:
        if self.proc_img is None:
            QMessageBox.warning(self, "Warning", "No processed image to save.")
            return
        fn, ft = QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG (*.jpg);;TIFF (*.tiff)")
        if not fn:
            return
        img = Image.fromarray(self.proc_img)
        fmt = "TIFF" if ft.startswith("TIFF") else "JPEG"
        img.save(fn, format=fmt)
        QMessageBox.information(
            self,
            "Success",
            f"Image saved successfully to:\n{fn}"
        )


def main() -> None:
    app = QApplication(sys.argv)
    w = SolStretchProApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()