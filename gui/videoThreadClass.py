from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from faceDetector import detect
import time

class VideoThread(QThread):
    change_pixmap_signal_face = pyqtSignal(np.ndarray)
    change_pixmap_signal_frame = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        # capture from web cam
        self.cap = cv2.VideoCapture(0)
        self.last_frame = None
        self.freeze_face_flag = False
        self.time = 0

    def run(self):
        while self._run_flag:
            res,frame = self.cap.read()
            self.change_pixmap_signal_frame.emit(frame)
            if res:
                detected_face = detect(frame)
                if detected_face is not None and not self.check_freeze_face_flag():
                    self.change_pixmap_signal_face.emit(detected_face)
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits ret, warped_face for thread to finish"""
        self._run_flag = False
        self.wait()

    def resume(self):
        self._run_flag = True
        self.start()
    
    def save_frame(self):
        res, frame = self.cap.read()
        if res:
            return frame
    
    def freeze_face(self):
        self.freeze_face_flag = True
    
    def unfreeze_face(self):
        self.freeze_face_flag = False
    
    def check_freeze_face_flag(self):
        if self.time == 0:
            self.unfreeze_face()
            return self.freeze_face_flag
        elif self.time + 10 > time.time():
            self.freeze_face()
        else:
            self.unfreeze_face() 
            self.last_frame = None
        return self.freeze_face_flag
        
        

