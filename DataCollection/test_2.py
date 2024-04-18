from playsound import playsound
import time
from threading import Thread

PAUSE_DURATION = 1

def start_audio_thread(file_idx: int):
    n_thread = Thread(target=play_audio, args=(file_idx,))
    n_thread.start()

def start_countdown_thread():
    n_thread = Thread(target=start_countdown)
    n_thread.start()

def start_pin_thread(pin: str):
    n_thread = Thread(target=handle_pin, args=(pin,))
    n_thread.start()

def play_audio(file_idx: int):
    file_name = f"./sounds/{file_idx}.wav"
    playsound(file_name)

def handle_pin(pin: str):
    for num in pin:
        num = int(num)
        start_audio_thread(num)
        time.sleep(PAUSE_DURATION)
          
def start_countdown():
    for i in range(5):
        file_idx = (5 - i)
        start_audio_thread(file_idx)
        time.sleep(1)

start_countdown_thread()

time.sleep(6)

start_pin_thread("235687")