# -*- coding: utf-8 -*-
import os
from PyQt5.QtCore  import Qt, QPropertyAnimation, QRect, QEasingCurve, QTimer
from PyQt5.QtGui   import (QPixmap, QFont, QPainter, QLinearGradient,
                            QColor, QPen, QBrush, QRadialGradient)
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QPushButton,
                              QSpacerItem, QSizePolicy, QGraphicsDropShadowEffect)


class SplashScreen(QWidget):
    def __init__(self, on_start_callback, resource_path):
        super().__init__()
        self.on_start_callback = on_start_callback
        self.resource_path     = resource_path

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setFixedSize(900, 520)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        # Botón minimizar: la portada es frameless (sin barra de título del
        # SO), así que no tiene minimizar/cerrar nativos — se agrega uno
        # propio en la esquina superior derecha para poder minimizarla
        # mientras queda esperando el clic en "Iniciar aplicación".
        self.btn_min = QPushButton("─", self)
        self.btn_min.setGeometry(900 - 44, 12, 32, 28)
        self.btn_min.setCursor(Qt.PointingHandCursor)
        self.btn_min.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,30);
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background: rgba(255,255,255,70); }
            QPushButton:pressed { background: rgba(255,255,255,100); }
        """)
        self.btn_min.setToolTip("Minimizar")
        self.btn_min.clicked.connect(self.showMinimized)
        self.btn_min.raise_()

        # Intentar cargar imagen de fondo si existe
        self._bg_pixmap = None
        for candidate in [
            resource_path(os.path.join("Menu", "fondo_optimizacion.png")),
            resource_path(os.path.join("Menu", "Fondo", "Imgen de fondo.jpg")),
            resource_path(os.path.join("Menu", "Fondo", "Imgen de fondo.png")),
            resource_path("fondo_optimizacion.png"),
        ]:
            if candidate and os.path.exists(candidate):
                px = QPixmap(candidate)
                if not px.isNull():
                    self._bg_pixmap = px
                    break

        # Layout principal
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(0)

        # Título
        self.title = QLabel("OPTIMIZADOR DE FUNCIONES", self)
        self.title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.title.setStyleSheet("color: white; background: transparent;")
        f = QFont("Segoe UI", 26, QFont.Bold)
        self.title.setFont(f)
        shadow_t = QGraphicsDropShadowEffect()
        shadow_t.setBlurRadius(28)
        shadow_t.setOffset(0, 4)
        shadow_t.setColor(QColor(0, 0, 0, 230))
        self.title.setGraphicsEffect(shadow_t)
        layout.addWidget(self.title)

        # Subtítulo
        self.subtitle = QLabel("Métodos Clásicos  ·  Búsqueda de Línea  ·  Metaheurísticos", self)
        self.subtitle.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.subtitle.setStyleSheet(
            "color: rgba(220,240,255,240); background: transparent; letter-spacing: 1px;"
            "text-shadow: 0px 2px 8px rgba(0,0,0,200);")
        self.subtitle.setFont(QFont("Segoe UI", 10))
        layout.addWidget(self.subtitle)

        layout.addItem(QSpacerItem(20, 18, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # Separador decorativo
        self.sep = QLabel(self)
        self.sep.setFixedSize(180, 2)
        self.sep.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 transparent, stop:0.5 rgba(100,180,255,200), stop:1 transparent);"
            "border: none;")
        layout.addWidget(self.sep, alignment=Qt.AlignHCenter)

        layout.addItem(QSpacerItem(20, 180, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Botón
        self.btn = QPushButton("  ▶  Iniciar aplicación", self)
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setFixedHeight(48)
        self.btn.setFixedWidth(280)
        self.btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #1a7fe8, stop:1 #0fb8c9);
                color: white;
                border: none;
                border-radius: 24px;
                font-size: 14px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #2e91f0, stop:1 #1fd0e0);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #1060b8, stop:1 #0a90a0);
            }
        """)
        shadow_b = QGraphicsDropShadowEffect()
        shadow_b.setBlurRadius(22)
        shadow_b.setOffset(0, 5)
        shadow_b.setColor(QColor(20, 100, 200, 140))
        self.btn.setGraphicsEffect(shadow_b)
        layout.addWidget(self.btn, alignment=Qt.AlignHCenter)

        layout.addItem(QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # Pie
        self.lbl_ver = QLabel("v2.0  ·  Python · PyQt5 · Matplotlib", self)
        self.lbl_ver.setAlignment(Qt.AlignHCenter)
        self.lbl_ver.setStyleSheet(
            "color: rgba(160,200,240,130); font-size: 10px; background: transparent;")
        layout.addWidget(self.lbl_ver)

        self.anim = QPropertyAnimation(self.btn, b"geometry")
        self.anim.setDuration(700)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.btn.clicked.connect(self._start)
        self._started = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        if self._bg_pixmap:
            scaled = self._bg_pixmap.scaled(
                w, h,
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation
            )
            # Centrar el recorte si la imagen es más grande que el widget
            x_off = (scaled.width()  - w) // 2
            y_off = (scaled.height() - h) // 2
            painter.drawPixmap(0, 0, scaled, x_off, y_off, w, h)
            # Degradado sutil solo en la zona inferior para legibilidad del texto
            from PyQt5.QtGui import QLinearGradient as _LG
            bottom_grad = _LG(0, h * 0.72, 0, h)
            bottom_grad.setColorAt(0.0, QColor(0, 0, 0, 0))
            bottom_grad.setColorAt(1.0, QColor(4, 14, 42, 140))
            painter.fillRect(0, int(h * 0.72), w, int(h * 0.28), QBrush(bottom_grad))
        else:
            grad = QLinearGradient(0, 0, w, h)
            grad.setColorAt(0.0,  QColor(4,  14,  42))
            grad.setColorAt(0.45, QColor(8,  28,  75))
            grad.setColorAt(1.0,  QColor(6,  50,  90))
            painter.fillRect(0, 0, w, h, QBrush(grad))

            radial = QRadialGradient(w * 0.5, h * 0.38, h * 0.65)
            radial.setColorAt(0.0, QColor(30, 100, 200, 55))
            radial.setColorAt(1.0, QColor(0,   0,   0,  0))
            painter.fillRect(0, 0, w, h, QBrush(radial))

            pen = QPen(QColor(40, 80, 160, 30), 1)
            painter.setPen(pen)
            step = 40
            for x in range(0, w + step, step):
                painter.drawLine(x, 0, x, h)
            for y in range(0, h + step, step):
                painter.drawLine(0, y, w, y)

        painter.end()

    def showEvent(self, event):
        super().showEvent(event)
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width()  - self.width())  // 2,
                  (screen.height() - self.height()) // 2)
        QTimer.singleShot(100, self._animate_btn)
        # Sin auto-arranque: la portada debe quedarse en pantalla hasta que
        # el usuario pulse "Iniciar aplicación" — lo único que se optimizó
        # fue que la portada APAREZCA rápido tras el doble clic (el import
        # pesado de main_window está diferido a iniciar_interfaz(), fuera
        # de este archivo, y solo se dispara al presionar el botón).

    def _animate_btn(self):
        g = self.btn.geometry()
        start = QRect(g.x(), g.y() + 30, g.width(), g.height())
        self.anim.setStartValue(start)
        self.anim.setEndValue(g)
        self.anim.start()

    def _start(self):
        if self._started:
            return
        self._started = True
        self.close()
        if callable(self.on_start_callback):
            self.on_start_callback()
