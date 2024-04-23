import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import video_recorder
import aux
import data_handler
import os
from preview_window import PreviewWindow
import cv2


header = ["file_name","pin"]

BASE_DIR = "./PinData/"

os.makedirs(BASE_DIR, exist_ok=True)

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.vc = cv2.VideoCapture(2)

        self.setWindowTitle("Data Collector")

        video_recorder.init_variables(BASE_DIR, self)

        data_handler.init_variables(header, self)
        data_handler.init_csv()
        
        video_recorder.add_recording_ui()
        
        self.preview = PreviewWindow(self, self.vc)
        self.preview.show()

    def alert(self, text):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Bad Idx")
        dlg.setText(f"Invalid Index: {text}")
        button = dlg.exec()

        


app = QApplication(sys.argv)

window = MainWindow()
window.show()


app.exec()
data_handler.save_csv()
video_recorder.is_recording = False
video_recorder.is_recording_all = False