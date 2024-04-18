import numpy as np
import cv2
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import aux
from skimage import util, data, transform
from skimage.io import imsave
from scipy import ndimage



bb_img = cv2.imread("./bb_court/court_3.png")
SECTION_SIZE = 64

SECTION_SIZE_HEIGHT = 64
SECTION_SIZE_WIDTH = 64

MAX_ROW = int(bb_img.shape[0] / SECTION_SIZE_HEIGHT) - 1
MAX_COL = int(bb_img.shape[1] / SECTION_SIZE_WIDTH) - 1 

DISPLAY_DIR = "./bb_court/display.jpg"


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized



class BB_Window(QWidget):
    
    def __init__(self):
        super().__init__()
        self.setLayout(self.create_bb_court())
        self.add_grid_lines()
        self.bb_img = bb_img
        self.SECTION_SIZE = SECTION_SIZE
        self.MAX_ROW = MAX_ROW
        self.MAX_COL = MAX_COL
        self.DISPLAY_DIR = DISPLAY_DIR


    def init_variables(self, n_app:QMainWindow):
        app = n_app


    def highlight_court(self, row, col):
        start_row = int(row * SECTION_SIZE_HEIGHT)
        end_row = int((row+1) * SECTION_SIZE_HEIGHT)

        start_col = int(col * SECTION_SIZE_WIDTH)
        end_col = int((col+1) * SECTION_SIZE_WIDTH)
        
        n_img = bb_img.copy()
        
        n_img[start_row:end_row+1, start_col:end_col+1] = (0,0,255)
        n_img = image_resize(n_img, height=768)
        

        cv2.imwrite(DISPLAY_DIR, n_img)

        self.update_bb_court()

    def create_bb_court(self):
        layout = QVBoxLayout()

        self.bb_court_label = QLabel()

        self.bb_court_pic = QPixmap(DISPLAY_DIR)
        self.bb_court_label.setPixmap(self.bb_court_pic)

        layout.addWidget(self.bb_court_label)

        return layout


    def update_bb_court(self):
        self.bb_court_pic = QPixmap(DISPLAY_DIR)
        self.bb_court_label.setPixmap(self.bb_court_pic)
        

    def add_grid_lines(self):
        for i in range(0, len(bb_img), SECTION_SIZE_HEIGHT):
            for j in range(0, len(bb_img[i])):
                bb_img[i,j] = (0,0,0)

        for i in range(0, len(bb_img)):
            for j in range(0, len(bb_img[i]), SECTION_SIZE_WIDTH):
                bb_img[i,j] = (0,0,0)
