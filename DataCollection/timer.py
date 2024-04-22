from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from aux import *
from threading import Thread



def update_timer():
    app.time_remaining -= 100

    if app.time_remaining <= 999:
        app.timer.stop()
        print("Done")
    else:
        app.timer.start()
        
    
    update_label()

def update_label():
    secs_left = (app.time_remaining // 1000)%60
    minutes_left = (secs_left // (1000*60))%60

    app.label.setText(f"{minutes_left:02}:{secs_left:02}")


def apply_timer_screen(n_app, duration):
    global app
    app = n_app

    layout = QVBoxLayout()

    app.label = QLabel()

    app.timer = QTimer()
    app.timer.setSingleShot(False)
    app.timer.setInterval(100)

    app.timer.timeout.connect(update_timer)
    app.time_remaining = (duration*1000) + 1000

    update_label()

    app.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(app.label)

    change_layout(app, layout)