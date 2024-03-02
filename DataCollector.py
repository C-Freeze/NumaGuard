from threading import Thread
import cv2, time, sys
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QLineEdit, QComboBox, QPushButton, QHBoxLayout
    )
from PyQt5.QtGui import QImage, QPixmap

class DataEntry(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 300, 220)
        self.setWindowTitle('Thread')

        self.person_label = QLabel("Person Name:")
        # drop down list
        x = ["Carter", "Cody", "Tahlia", "Tyler"]
        self.person_edit = QComboBox()
        self.person_edit.addItems(x)

        self.sequence_label = QLabel("Sequence:")
        self.sequence_edit = QLineEdit()

        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;")

        # self.start_button.clicked.connect(self.start_thread)

        self.start_button.clicked.connect(
            lambda: print(self.person_edit.currentText(), self.sequence_edit.text())
        )

        self.exit_button = QPushButton("Exit")
        self.exit_button.setStyleSheet("background-color: #f44336; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;")
        # self.exit_button.clicked.connect(sys.exit)
        def exit_app():
            self.vw.close()
            sys.exit()
        self.exit_button.clicked.connect(exit_app)

        vbox = QVBoxLayout()
        vbox.addWidget(self.person_label)
        vbox.addWidget(self.person_edit)
        vbox.addWidget(self.sequence_label)
        vbox.addWidget(self.sequence_edit)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.exit_button)

        self.setLayout(vbox)
        self.vw = VideoWindow()
        self.vw.show()
        self.show()

class VideoStreamWidget(QWidget):
    def __init__(self, src=0):
        super().__init__()
        self.video_stream = cv2.VideoCapture(src)
        self.video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.video_stream.set(cv2.CAP_PROP_FPS, 30)
        self.video_stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.initUI()
        self.recording = False

    def update(self):
        while True:
            ret, frame = self.video_stream.read()
            if ret:
                self.display_frame(frame)
            else:
                break

    def show_frame(self):
        ret, frame = self.video_stream.read()
        if ret:
            self.display_frame(frame)
            
    def initUI(self):
        self.video_frame = QLabel()
        self.video_frame.setFixedSize(320, 240)
        self.video_frame.setStyleSheet("border: 1px solid black")
        
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.video_frame)
        self.setLayout(self.hbox)
        
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320, 240))
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Feeds")
        self.setGeometry(400, 100, 500, 400)
        self.initUI()
        
    def initUI(self):
        self.vbox = QVBoxLayout()
        
        self.streams = [VideoStreamWidget(i) for i in range(2)]

        for stream in self.streams:
            self.vbox.addWidget(stream)
        
        self.setLayout(self.vbox)
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = DataEntry()
    sys.exit(app.exec_())
