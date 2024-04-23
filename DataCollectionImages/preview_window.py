import cv2
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import sys
from PyQt6.QtMultimediaWidgets import *
from PyQt6.QtMultimedia import *


SECTION_SIZE_HEIGHT = 80
SECTION_SIZE_WIDTH = 80
FONT_SIZE = 0.7
THICKNESS = 2



def add_grid_lines(img):
    for i in range(0, len(img), SECTION_SIZE_HEIGHT):
        for j in range(0, len(img[i])):
            img[i,j] = (0,0,255)
            

    for i in range(0, len(img)):
        for j in range(0, len(img[i]), SECTION_SIZE_WIDTH):
            img[i,j] = (0,0,255)
        
    
    col = 0
    for i in range(0, len(img), SECTION_SIZE_WIDTH):

        row = 0
        for j in range(0, len(img[i]), SECTION_SIZE_HEIGHT):
            
            text = f"{row}, {col}"
            cv2.putText(img,text, (i, j + 15), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255,0,0), thickness=THICKNESS)
            row += 1

        col += 1



class PreviewWindow(QWidget):
    def __init__(self, n_app:QMainWindow, vc):
        super().__init__()

        global app

        app = n_app
        self.cap = vc

        self.setWindowTitle("Preview Window")
        
        self.label = QLabel('No Camera Feed')
        self.layout = QVBoxLayout()
        # layout.addWidget(button)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.displayFrame)
        self.timer.setInterval(100)
        self.timer.start()
        


    def displayFrame(self):
        try:
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            add_grid_lines(frame)
            frame = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)   
            self.label.setPixmap(QPixmap.fromImage(frame))
        except:
            print("Window is closed!")
        