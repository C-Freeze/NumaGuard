from threading import Thread
import cv2, time, sys
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QLineEdit, QComboBox, QPushButton, QHBoxLayout, QGridLayout, QScrollArea, QListWidget, QFileDialog
    )
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import os
import pyaudio
import wave
import sys

# length of data to read.
chunk = 1024
threads = []

'''
************************************************************************
      This is the start of the "minimum needed to read a wave"
************************************************************************
'''
    

class DataEntry(QWidget):
    def __init__(self, num_webcams=4):
        super().__init__()
        self.num_webcams = num_webcams
        self.initUI()
        self.out_path = "./data/"
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        
        self.idx = 0

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.person_label = QLabel("Person Name:")
        x = ["Carter", "Cody", "Tahlia", "Tyler"]
        self.person_edit = QComboBox()
        self.person_edit.addItems(x)

        self.sequence_label = QLabel("Sequence:")
        self.sequence_edit = QLineEdit()

        self.start_button = QPushButton("Start")
        
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)
        
        self.start_button.clicked.connect(self.start_recording)
        self.start_button.clicked.connect(
            lambda: print(self.person_edit.currentText(), self.sequence_edit.text())
        )
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.clicked.connect(lambda: print(f'Recording saved to {self.out_path}'))
        self.stop_button.setEnabled(False)


        self.exit_button = QPushButton("Exit")
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
            }
            QPushButton:hover {
                background-color: #424242;
            }
            QPushButton:pressed {
                background-color: #212121;
            }
        """)
        
        self.exit_button.clicked.connect(self.exit_app)

        self.video_frames = []
        self.video_streams = []
        self.recordings = []
        self.outs = []

        for i in range(self.num_webcams):
            video_frame = QLabel()
            video_frame.setFixedSize(320, 240)
            video_frame.setStyleSheet("border: 1px solid black")
            self.video_frames.append(video_frame)

            video_stream = cv2.VideoCapture(i)
            video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            video_stream.set(cv2.CAP_PROP_FPS, 30)
            video_stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            self.video_streams.append(video_stream)

            self.recordings.append(False)
            self.outs.append(None)

        self.recording_label = QLabel("Recording: OFF")
        self.recording_label.setAlignment(Qt.AlignCenter)
        self.recording_label.setStyleSheet("color: red; font-weight: bold")
        
        self.load_button = QPushButton("Load Sequences")
        self.load_button.clicked.connect(self.load_sequences)

        self.sequence_list = QListWidget()
        self.sequence_list.itemClicked.connect(self.update_sequence)

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.previous_sequence)
        self.prev_button.setEnabled(False)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_sequence)
        self.next_button.setEnabled(False)

        grid_layout = QGridLayout()
        for i in range(self.num_webcams):
            row = i // 2
            col = i % 2
            grid_layout.addWidget(self.video_frames[i], row, col)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setLayout(grid_layout)
        scroll_area.setWidget(scroll_content)

        vbox = QVBoxLayout()
        vbox.addWidget(self.person_label)
        vbox.addWidget(self.person_edit)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.stop_button)
        vbox.addWidget(self.exit_button)
        vbox.addWidget(scroll_area)
        vbox.addWidget(self.recording_label)

        vbox.addWidget(self.load_button)
        vbox.addWidget(self.sequence_list)
        vbox.addWidget(self.prev_button)
        vbox.addWidget(self.next_button)
        vbox.addWidget(self.sequence_label)
        vbox.addWidget(self.sequence_edit)
        self.setLayout(vbox)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)
        
    
        
        
    def load_sequences(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt)")
        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]
            self.load_sequence_file(selected_file)

    def load_sequence_file(self, file_path):
        self.sequence_list.clear()
        self.sequences = []
        with open(file_path, 'r') as file:
            for line in file:
                self.sequences.append(line.strip())
        self.sequence_list.addItems(self.sequences)
        self.current_index = 0
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(len(self.sequences) > 1)

    def update_sequence(self, item):
        self.current_index = self.sequence_list.currentRow()
        self.sequence_edit.setText(item.text())

    def previous_sequence(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.sequence_list.setCurrentRow(self.current_index)
            self.sequence_edit.setText(self.sequences[self.current_index])
            self.next_button.setEnabled(True)
            if self.current_index == 0:
                self.prev_button.setEnabled(False)

    def next_sequence(self):
        if self.current_index < len(self.sequences) - 1:
            self.current_index += 1
            self.sequence_list.setCurrentRow(self.current_index)
            self.sequence_edit.setText(self.sequences[self.current_index])
            self.prev_button.setEnabled(True)
            if self.current_index == len(self.sequences) - 1:
                self.next_button.setEnabled(False)

    def update_frames(self):
        for i in range(self.num_webcams):
            ret, frame = self.video_streams[i].read()
            if ret:
                if self.recordings[i]:
                    self.outs[i].write(frame)
                self.display_frame(frame, i)

    def display_frame(self, frame, index):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320, 240))
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.video_frames[index].setPixmap(pix)

    def start_recording(self):
        person = self.person_edit.currentText()
        sequence = self.sequence_edit.text()

        for i in range(self.num_webcams):
            if not self.recordings[i]:
                self.recordings[i] = True
                filename = f"{person}_{sequence}_cam{i+1}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.outs[i] = cv2.VideoWriter(self.out_path + filename, fourcc, 30, (320, 240))

        self.recording_label.setText("Recording: ON")
        self.recording_label.setStyleSheet("color: green; font-weight: bold")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_recording(self):
        for i in range(self.num_webcams):
            if self.recordings[i]:
                self.recordings[i] = False
                self.outs[i].release()
                self.outs[i] = None

        self.recording_label.setText("Recording: OFF")
        self.recording_label.setStyleSheet("color: red; font-weight: bold")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def exit_app(self):
        self.stop_recording()
        for video_stream in self.video_streams:
            video_stream.release()
        self.close()
        sys.exit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = DataEntry()
    ex.show()
    sys.exit(app.exec_())