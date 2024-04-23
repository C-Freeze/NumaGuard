from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import threading
import os
import random

threads = []




def change_layout(app, layout):
    app.main_widget = QWidget()
    app.main_widget.setLayout(layout)
    app.setCentralWidget(app.main_widget)


def get_folder_size(dir):
    _, _, files = next(os.walk(dir))
    file_count = len(files)
    return file_count


# record_length = get_folder_size("/home/tyler/Documents/Data/HorseData/HorseData1/")

# print(f"Record Length is: {record_length}")


