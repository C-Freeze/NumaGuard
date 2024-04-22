import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import video_recorder
import aux
import record_handler
import os


header = ["file_name","pin"]

BASE_DIR = "./PinData/"

os.makedirs(BASE_DIR, exist_ok=True)

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Data Collector")

        # self.setStyleSheet("QLabel{font-size: 12pt;} QPushButton{font: 12px;}")

        video_recorder.init_variables(BASE_DIR, 10, self)

        record_handler.init_variables(header, self)
        record_handler.init_csv()
        
        video_recorder.add_recording_ui()

        # video_recorder.start_all_frame_recording_thread()

    def alert(self, text):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Bad Idx")
        dlg.setText(f"Invalid Index: {text}")
        button = dlg.exec()

        


app = QApplication(sys.argv)

window = MainWindow()
window.show()


app.exec()
record_handler.save_csv()
video_recorder.is_recording = False
video_recorder.is_recording_all = False
