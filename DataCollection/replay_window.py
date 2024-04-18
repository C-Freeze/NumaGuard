import numpy as np
import cv2
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtMultimediaWidgets import *
from PyQt6.QtMultimedia import *
import aux


class Replay_Window(QWidget):
    
    def __init__(self, file_name, base_dir):
        super().__init__()

        self.setGeometry(0, 0, 1024, 768)

        self.setWindowTitle(f"{file_name}.avi")

        self.player = QMediaPlayer()
        self.player.setSource(QUrl.fromLocalFile(f"{file_name}"))

        self.player.setLoops(-1)
        
        videoWidget = QVideoWidget()
        self.player.setVideoOutput(videoWidget)
        

        layout = QVBoxLayout()
        layout.addWidget(videoWidget)

        
        self.setLayout(layout)

        self.player.play()

    