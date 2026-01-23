# -*- coding: utf-8 -*-
import os
from PyQt5.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QSpacerItem, QSizePolicy


class SplashScreen(QWidget):
    def __init__(self, on_start_callback, resource_path):
        super().__init__()
        self.on_start_callback = on_start_callback
        self.resource_path = resource_path

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setFixedSize(900, 520)

        # Fondo
        self.bg = QLabel(self)
        self.bg.setGeometry(0, 0, self.width(), self.height())
        self.bg.setScaledContents(True)

        img_path = self.resource_path(os.path.join("Menu", "fondo_optimizacion.png"))
        if os.path.exists(img_path):
            self.bg.setPixmap(QPixmap(img_path))

        # Capa layout encima
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(12)

        # Título
        self.title = QLabel("OPTIMIZADOR DE FUNCIONES", self)
        self.title.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.title.setStyleSheet("color: white;")
        f = QFont()
        f.setPointSize(24)
        f.setBold(True)
        self.title.setFont(f)
        layout.addWidget(self.title)

        layout.addItem(QSpacerItem(20, 260, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Botón centrado
        self.btn = QPushButton("Iniciar aplicación", self)
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setFixedHeight(44)
        self.btn.setFixedWidth(260)
        self.btn.setStyleSheet("""
            QPushButton{
                background-color: #2EA8FF;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton:hover{ background-color: #49B3FF; }
            QPushButton:pressed{ background-color: #1F8EDB; }
        """)
        layout.addWidget(self.btn, alignment=Qt.AlignHCenter)

        # Animación suave del botón (entrada)
        self.anim = QPropertyAnimation(self.btn, b"geometry")
        self.anim.setDuration(650)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)

        self.btn.clicked.connect(self._start)

    def showEvent(self, event):
        super().showEvent(event)
        # animar botón desde abajo
        g = self.btn.geometry()
        start = QRect(g.x(), g.y()+25, g.width(), g.height())
        end = QRect(g.x(), g.y(), g.width(), g.height())
        self.anim.setStartValue(start)
        self.anim.setEndValue(end)
        self.anim.start()

    def _start(self):
        self.close()
        if callable(self.on_start_callback):
            self.on_start_callback()
