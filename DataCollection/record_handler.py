import pandas as pd
import numpy as np
import torch
import random
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import const
import aux
import video_recorder

records = []
idx = 0

dribble_count = 0

atmpt_loc_x = 0
atmpt_loc_y = 0

shooter_id = 0
passer_id = 0

self_lob = False
atmpt_type = const.SHOT_ATTEMPT

max_x = 20
max_y = 20


pin_file = open("./pins.txt", "r+")
pins = pin_file.read()
pins = pins.split("\n")
pin_length = len(pins)

current_pin = "0000"



def get_random_pin():
    global current_pin

    # idx = random.randrange(0,pin_length)

    # current_pin = pins[idx]
    
    current_pin = idx // 10

    app.actual_pin_text.setText(f"{current_pin}")


def init_variables(n_header, n_app:QMainWindow):
    global app
    global header
    app = n_app
    header = n_header

    print(f"\nRecords Header\n=============\n{header}\n=============\n")


def create_labels():
    app.pin_label = QLabel("Pin")
    app.actual_pin_text = QLabel(f"{current_pin}")

    app.get_new_pin_button = QPushButton("Get New Pin")

    app.get_new_pin_button.clicked.connect(get_random_pin)

    pin_layout = QHBoxLayout()
    pin_layout.addWidget(app.pin_label)
    pin_layout.addWidget(app.actual_pin_text)

    parent_layout = QVBoxLayout()

    parent_layout.addLayout(pin_layout)
    parent_layout.addWidget(app.get_new_pin_button)

    return parent_layout




def record_data(file_name):
    global idx
    global app
    
    if idx >= len(records):
        records.append([file_name, current_pin])
    else:
        records[idx] = [file_name, current_pin]


    print(records[idx])
    idx += 1

    app.idx = idx

    video_recorder.update_spin_box()
    
    save_csv()


def save_idx():
    global idx
    with open("./idx.txt", "w") as f:
        f.write(idx)
        f.close()


def save_csv():
    df = pd.DataFrame(records, columns=header)
    df.to_csv("./data.csv", index=False)

def init_csv():
    try:
        global records
        global idx
        df = pd.read_csv("./data.csv", header=0, dtype=str)
        records = df.to_numpy().tolist()
        idx = len(records)
    except:
        print("ded")
        return