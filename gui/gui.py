"""
compilar la ui cuando se hace algun cambio

> pyuic5  -x mainWindow.ui > ui_mainWindow.py   


borrar estas lineas del archivo ui_mainWindow.py:

        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout = QtWidgets.QGridLayout(MainWindowDlg)

agregar estas 3 líneas al archivo ui_mainWindow.py

        self.centralwidget = QtWidgets.QWidget()
        MainWindowDlg.setCentralWidget(self.centralwidget)
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)

"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import sys
from videoThreadClass import VideoThread
import ui_mainWindow
import time

qtCreatorFile = "mainWindow.ui" # Enter file here.
########################################################################   
class mainWindow(QtWidgets.QMainWindow,ui_mainWindow.Ui_MainWindowDlg):
    def __init__(self, parent=None):
        
        super(mainWindow, self).__init__(parent)
        
        ###############################################################################
        print('Inicializando GUI')
        
        self.setupUi(self)
        
        # connect the buttons
        self.confirmButton.clicked.connect(self.capturar_foto)
        self.saveButton.clicked.connect(self.guardar_imagen)
        ###############################################################################
        # launch demo thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal_face.connect(self.update_image_face)
        self.thread.change_pixmap_signal_frame.connect(self.update_image_frame)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
    
    @pyqtSlot(np.ndarray)
    def update_image_face(self, cv_img):
        self.update_image(self.faceView, cv_img, (self.faceView.width(), self.faceView.height()))
    
    @pyqtSlot(np.ndarray)
    def update_image_frame(self, cv_img):
        self.update_image(self.imageView, cv_img, (self.imageView.width(), self.imageView.height()))

    @pyqtSlot(np.ndarray)
    def update_image(self, label, cv_img, size):
        """Updates the face_label with a new opencv image"""
        qt_img = self.convert_cv_qt(size, cv_img)
        label.setPixmap(qt_img)

    def convert_cv_qt(self,size, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        x, y = size
        p = convert_to_Qt_format.scaled(x, y, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def capturar_foto(self):
        if self.thread.last_frame is None:
            self.thread.last_frame = self.thread.save_frame()
            self.thread.time = time.time()
            self.thread.freeze_face()
            self.statusBar().showMessage('Foto capturada!', 3000)
    
    def guardar_imagen(self):
        path = '/home/juli/Desktop/trabajo/stand-sdc/remote/hyperstyle/img'
        if self.thread.last_frame is not None:
            current_time = time.time().__str__().replace('.','-')
            cv2.imwrite(f'{path}/{current_time}.png',self.thread.last_frame)
            self.thread.last_frame = None
            self.thread.unfreeze_face()
            print('Imagen guardada')
            self.statusBar().showMessage('Imagen guardada!', 3000)
        else:
            print('No hay imagen para guardar')
            self.statusBar().showMessage('No hay imagen para guardar', 3000)


##########################################################################        
##########################################################################        
    
    
    def createScene(self,pixmap_original,H,W,factor=0.98):
        scene = None
        w = pixmap_original.width()
        h = pixmap_original.height()

        try:
            scene = QtWidgets.QGraphicsScene()
        except OSError as err:
            print("OS error: {0}".format(err))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        pixmap = pixmap_original.copy()
        pxitem = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(pxitem)       
        
        if factor*H/h < factor*W/w:
            scale = factor*H/h
        else:
            scale = factor*W/w

        for it in scene.items():             
             it.setScale(scale)
        return scene
    

        
##########################################################################        
##########################################################################                       
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())