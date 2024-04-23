import cv2
import time
import threading
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import aux
import data_handler
import replay_window
from playsound import playsound
import numpy as np
import time
import datetime
import uuid


threads = []
is_recording = False
is_recording_all = True
has_recorded = False
is_looping = False
RECORDING_SECONDS = 3
RECORD_DELAY = 3
all_frames = []

def init_variables(base_dir, n_app:QMainWindow):
    global vc
    global WEBCAM_HEIGHT
    global WEBCAM_WIDTH
    global BASE_DIR
    global RECORD_SECONDS
    global recording_counter_secs
    global recording_counter_mins

    global app
    app = n_app

    vc = app.vc
    

    # master_frame = np.zeros((1225,25))

    grabbed, master_frame = vc.read()
    master_frame = master_frame.shape

    WEBCAM_WIDTH = master_frame[1]
    WEBCAM_HEIGHT = master_frame[0]
    BASE_DIR = base_dir
    recording_counter_secs = 0
    recording_counter_mins = 0

    app.BASE_DIR = base_dir

    
    


def update_is_looping():
    global is_looping
    is_looping = app.is_looping.isChecked()


def start_sound_thread(file_name):
    n_thread = threading.Thread(target=play_ding_sound, args=(file_name,))
    n_thread.start()


def play_ding_sound(file_name):
    playsound(f"./sounds/{file_name}")


def update_spin_box():
    # app.current_idx_spin_box.setRange(0, aux.record_length + 100)
    app.current_idx_spin_box.setRange(0, len(data_handler.records))
    app.current_idx_spin_box.setValue(data_handler.idx)


def add_recording_ui():
    main_layout = QVBoxLayout()
    
    recording_layout = QHBoxLayout()


    # Recording Stuff
    app.record_btn = QPushButton("Start Recording")
    app.play_last_recording_btn = QPushButton("Play Recording At Index")

    app.record_btn.clicked.connect(recording_func)
    app.play_last_recording_btn.clicked.connect(replay_func)

    recording_layout.addWidget(app.record_btn)
    recording_layout.addWidget(app.play_last_recording_btn)


    # Recording Counter
    app.recording_counter_label = QLabel(f"{recording_counter_mins:02}:{recording_counter_secs:02}")
    app.recording_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    app.recording_counter_label.setStyleSheet("font-size: 12pt")


    # Index Spin Box
    app.current_idx_label = QLabel("Current Index")
    app.current_idx_spin_box = QSpinBox()
    update_spin_box()


     # Recording Duration and Delay Stuff
    app.recording_duration_label = QLabel("Recording Duration")
    app.recording_duration_spinbox = QSpinBox()

    app.recording_delay_label = QLabel("Recording Delay")
    app.recording_delay_spinbox = QSpinBox()

    duration_delay_label_layout = QHBoxLayout()
    duration_delay_label_layout.addWidget(app.recording_duration_label)
    duration_delay_label_layout.addWidget(app.recording_delay_label)

    duration_delay_spin_box_layout = QHBoxLayout()
    duration_delay_spin_box_layout.addWidget(app.recording_duration_spinbox)
    duration_delay_spin_box_layout.addWidget(app.recording_delay_spinbox)

    duration_delay_layout = QVBoxLayout()
    duration_delay_layout.addLayout(duration_delay_label_layout)
    duration_delay_layout.addLayout(duration_delay_spin_box_layout)

    app.recording_delay_spinbox.setValue(RECORD_DELAY)
    app.recording_duration_spinbox.setValue(RECORDING_SECONDS)

    app.recording_duration_spinbox.textChanged.connect(change_duration)
    app.recording_delay_spinbox.textChanged.connect(change_delay)


    # Is Looping Checkbox
    app.is_looping = QCheckBox("Will Record In A Loop")
    app.is_looping.stateChanged.connect(update_is_looping)
    app.current_idx_spin_box.textChanged.connect(change_idx)


    # Setting Stuff
    settings_layout = data_handler.create_labels()


    # Add all the stuff together
    main_layout.addLayout(settings_layout)
    main_layout.addWidget(app.current_idx_label)
    main_layout.addWidget(app.current_idx_spin_box)
    main_layout.addLayout(duration_delay_layout)
    main_layout.addWidget(app.is_looping)
    main_layout.addLayout(recording_layout)
    main_layout.addWidget(app.recording_counter_label)
    aux.change_layout(app, main_layout)


    # Recording Timer
    app.timer = QTimer()
    app.timer.setInterval(1000)
    app.timer.timeout.connect(update_recording_counter)


    # Init Pin
    data_handler.get_random_pin()


def get_new_file_name():
    return f"{BASE_DIR}{data_handler.current_pin}_{str(uuid.uuid4())}.jpeg"


def update_recording_counter():
    global recording_counter_secs
    global recording_counter_mins
    global is_recording
    recording_counter_secs += 1

    if recording_counter_secs == 60:
        recording_counter_secs = 0
        recording_counter_mins += 1

    if recording_counter_secs == RECORDING_SECONDS:
        app.recording_counter_label.setText(f"{recording_counter_mins:02}:{recording_counter_secs:02}")
        

        record_frame()

        start_sound_thread("ding.mp3")

        file_name = get_new_file_name()
        save_image(file_name)
        data_handler.record_data(file_name)
        reset_recording_counter()
        time.sleep(RECORD_DELAY)

        if is_looping:
            is_recording = True
            start_recording_counter()
        
        else:
            app.record_btn.setText("Start Recording")
            
    else:
        app.recording_counter_label.setText(f"{recording_counter_mins:02}:{recording_counter_secs:02}")


def change_idx():
    num = app.current_idx_spin_box.value()
    data_handler.idx = num


def change_duration():
    global RECORDING_SECONDS
    num = app.recording_duration_spinbox.value()
    RECORDING_SECONDS = num


def change_delay():
    global RECORD_DELAY
    num = app.recording_delay_spinbox.value()
    RECORD_DELAY = num


def replay_func():
    
    if data_handler.idx >= len(data_handler.records):
        print("Outside of index!!")
        return
    
    app.replay_window = replay_window.Replay_Window(data_handler.records[data_handler.idx][0], app.BASE_DIR)
    app.replay_window.show()



def reset_recording_counter():
    global recording_counter_secs
    global recording_counter_mins
    recording_counter_secs = 0
    recording_counter_mins = 0
    app.timer.stop()
    app.recording_counter_label.setText(f"{recording_counter_mins:02}:{recording_counter_secs:02}")
    app.recording_counter_label.setStyleSheet("color: black; font-size: 12pt;")
    


def start_recording_counter():
    data_handler.get_random_pin()
    app.recording_counter_label.setStyleSheet("color: red; font-size: 12pt;")
    app.timer.start()
    

def recording_func():
    
    global is_recording
    global recording_counter_secs
    global recording_counter_mins
    global all_frames

    if is_recording:
        is_recording = False
        app.record_btn.setText("Start Recording")
        reset_recording_counter()
        delete_video()

    else:
        print(f"Starting In {RECORD_DELAY} Seconds!")
        time.sleep(RECORD_DELAY)
        is_recording = True
        
        start_recording_counter()
        app.record_btn.setText("Delete Recording")
        


def start_frame_recording_thread():
    global threads
    global is_recording
    start_sound_thread("bingus.wav")
    
    n_thread = threading.Thread(target=record_frame)
    threads.append(n_thread)
    n_thread.start()
    



def save_image(file_name):
    global WEBCAM_HEIGHT
    global WEBCAM_WIDTH
    global image

    print("Saving image...")
    
    for thread in threads:
        thread.join()
        
    cv2.imwrite(file_name, image)

    print("Image Saved!")



def delete_video():
    global WEBCAM_HEIGHT
    global WEBCAM_WIDTH
    global all_frames

    print("Deleting video...")
    
    for thread in threads:
        thread.join()
    
    all_frames.clear()

    print("Video Deleted!")
    
    
    

def record_frame():
    global vc
    global WEBCAM_HEIGHT
    global WEBCAM_WIDTH
    global BASE_DIR
    global is_recording
    global is_looping
    global image
    file_name = data_handler.idx
    
    
    rval, image = vc.read()
