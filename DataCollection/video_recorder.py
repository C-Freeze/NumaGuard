import cv2
import time
import threading
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import aux
import record_handler
import replay_window
from playsound import playsound
import numpy as np
import time
import datetime


threads = []
is_recording = False
is_recording_all = True
has_recorded = False
is_looping = False
RECORDING_SECONDS = 8
all_frames = []

def init_variables(base_dir, record_seconds, n_app:QMainWindow):
    global vc
    global WEBCAM_HEIGHT
    global WEBCAM_WIDTH
    global BASE_DIR
    global RECORD_SECONDS
    global recording_counter_secs
    global recording_counter_mins

    global app
    app = n_app

    vc = cv2.VideoCapture(2)

    # master_frame = np.zeros((1225,25))

    grabbed, master_frame = vc.read()
    master_frame = master_frame.shape

    WEBCAM_WIDTH = master_frame[1]
    WEBCAM_HEIGHT = master_frame[0]
    BASE_DIR = base_dir
    RECORD_SECONDS = record_seconds
    recording_counter_secs = 0
    recording_counter_mins = 0

    app.BASE_DIR = base_dir
    


def update_is_looping():
    global is_looping
    is_looping = app.is_looping.isChecked()

def play_ding_sound(file_name):
    playsound(f"./sounds/{file_name}")


def update_spin_box():
    # app.current_idx_spin_box.setRange(0, aux.record_length + 100)
    app.current_idx_spin_box.setRange(0, len(record_handler.records))
    app.current_idx_spin_box.setValue(record_handler.idx)

def add_recording_ui():
    main_layout = QVBoxLayout()
    
    recording_layout = QHBoxLayout()


    app.record_btn = QPushButton("Start Recording")
    app.play_last_recording_btn = QPushButton("Play Recording At Index")

    app.recording_counter_label = QLabel(f"{recording_counter_mins:02}:{recording_counter_secs:02}")
    app.recording_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    app.recording_counter_label.setStyleSheet("font-size: 12pt")

    app.current_idx_label = QLabel("Current Index")
    app.current_idx_spin_box = QSpinBox()
    update_spin_box()

    app.is_looping = QCheckBox("Will Record In A Loop")
    app.is_looping.stateChanged.connect(update_is_looping)

    app.current_idx_spin_box.textChanged.connect(change_idx)

    app.record_btn.clicked.connect(recording_func)
    app.play_last_recording_btn.clicked.connect(replay_func)

    recording_layout.addWidget(app.record_btn)
    
    recording_layout.addWidget(app.play_last_recording_btn)
    
    settings_layout = record_handler.create_labels()
    
    main_layout.addLayout(settings_layout)
    main_layout.addWidget(app.current_idx_label)
    main_layout.addWidget(app.current_idx_spin_box)
    main_layout.addWidget(app.is_looping)
    main_layout.addLayout(recording_layout)
    main_layout.addWidget(app.recording_counter_label)
    
    aux.change_layout(app, main_layout)


    app.timer = QTimer()
    app.timer.setInterval(1000)
    app.timer.timeout.connect(update_recording_counter)

    record_handler.get_random_pin()


def update_recording_counter():
    global recording_counter_secs
    global recording_counter_mins
    global is_recording
    recording_counter_secs += 1

    if recording_counter_secs == 60:
        recording_counter_secs = 0
        recording_counter_mins += 1

    if recording_counter_secs == RECORDING_SECONDS:
        
        print("Saving Video...")
        app.recording_counter_label.setText(f"Saving Video!")
        play_ding_sound("ding.mp3")
        time.sleep(3)
        file_name = f"{BASE_DIR}{record_handler.current_pin}_{get_time_stamp()}.avi"
        is_recording = False
        save_video(file_name)
        record_handler.record_data(file_name)
        reset_recording_counter()

        if is_looping:
            is_recording = True
            start_recording_counter()
            start_frame_recording_thread()
        
        else:
            app.record_btn.setText("Start Recording")
            
    else:
        app.recording_counter_label.setText(f"{recording_counter_mins:02}:{recording_counter_secs:02}")


def change_idx():
    num = app.current_idx_spin_box.value()
    record_handler.idx = num


def replay_func():
    
    if record_handler.idx >= len(record_handler.records):
        print("Outside of index!!")
        return
    
    app.replay_window = replay_window.Replay_Window(record_handler.records[record_handler.idx][0], app.BASE_DIR)
    app.replay_window.show()



def reset_recording_counter():
    global recording_counter_secs
    global recording_counter_mins
    recording_counter_secs = 0
    recording_counter_mins = 0
    app.timer.stop()
    app.recording_counter_label.setText(f"{recording_counter_mins:02}:{recording_counter_secs:02}")
    app.recording_counter_label.setStyleSheet("color: white; font-size: 12pt;")


def start_recording_counter():
    record_handler.get_random_pin()
    app.recording_counter_label.setStyleSheet("color: red; font-size: 12pt;")
    app.timer.start()

def get_time_stamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S%f')
    

def recording_func():
    
    global is_recording
    global recording_counter_secs
    global recording_counter_mins
    global all_frames

    if is_recording:
        is_recording = False
        app.record_btn.setText("Start Recording")
        reset_recording_counter()
        all_frames.clear()
        print("Recording Deleted")

    else:
        print("Starting In 3 Seconds!")
        time.sleep(3)
        is_recording = True
        start_frame_recording_thread()
        start_recording_counter()
        app.record_btn.setText("Delete Recording")
        
    

def shot_made_missed_func():
    global app
    global is_recording
    global is_looping

    if not is_recording:
        dlg = QDialog(app)
        dlg.setWindowTitle("Record A Video First!")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Record A Video First!"))
        dlg.setLayout(layout)
        dlg.exec()
        return

    is_recording = False
    save_video()
    record_handler.record_data()
    reset_recording_counter()

    if is_looping:
        is_recording = True
        start_recording_counter()
        start_frame_recording_thread()
    else:
        app.record_btn.setText("Start Recording")


    n_thread = threading.Thread(target=play_ding_sound, args=(6,))
    n_thread.start()


def start_frame_recording_thread():
    global threads
    global is_recording
    
    n_thread = threading.Thread(target=play_ding_sound, args=("bingus.wav",))
    n_thread.start()
    
    n_thread = threading.Thread(target=record_frames)
    threads.append(n_thread)
    n_thread.start()
    
    


def start_all_frame_recording_thread():
    global threads
    global is_recording
    n_thread = threading.Thread(target=record_all_frames)
    n_thread.start()


def save_video(file_name):
    global WEBCAM_HEIGHT
    global WEBCAM_WIDTH
    global all_frames
    
    for thread in threads:
        thread.join()
    
    writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"MJPG"), 30,(WEBCAM_WIDTH,WEBCAM_HEIGHT))
    
    for frame in all_frames:
        writer.write(frame)
        
    writer.release()
    all_frames.clear()
    
    
    

def record_frames():
    global vc
    global WEBCAM_HEIGHT
    global WEBCAM_WIDTH
    global BASE_DIR
    global RECORD_SECONDS
    global is_recording
    global is_looping
    global all_frames
    file_name = record_handler.idx
    
    
    if all_frames is None:
        all_frames = []

    start_time = time.time()
    
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while is_recording:
        rval, frame = vc.read()
        all_frames.append(frame)


def record_all_frames():
    global vc
    global WEBCAM_HEIGHT
    global WEBCAM_WIDTH
    global BASE_DIR
    global RECORD_SECONDS
    global is_recording_all
    file_name = record_handler.idx

    writer = cv2.VideoWriter(f"{BASE_DIR}full.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30,(WEBCAM_WIDTH,WEBCAM_HEIGHT))
    
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while is_recording_all:
        writer.write(frame)
        rval, frame = vc.read()

    writer.release()