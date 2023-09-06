import sys
sys.path.append('./FaceDetection')
import os
from mtcnn_pytorch.mtcnn import MTCNN
import numpy as np
from face_det_functions.align_trans import warp_and_crop_face, get_reference_facial_points
from copy import copy
import cv2
from pathlib import Path
from datetime import datetime


#############################################################################
REFERENCE_CROP = 112.
crop_size = 1024 # specify size of aligned faces, align and crop with padding
scale = crop_size / REFERENCE_CROP
reference = get_reference_facial_points(ref_pts='stylegan', default_square = True) * scale
mtcnn_pt = MTCNN(min_face_size=100, device='cpu')
#############################################################################

def detect(frame):
    #deteccion y croppeo de rostro
    bboxes, scores, landmarks = mtcnn_pt.detect(frame,landmarks=True)
    if bboxes is not None and len(bboxes) > 0:
        points = landmarks[0]
        warped_face = warp_and_crop_face(np.array(frame), points, reference, crop_size=(crop_size, crop_size))
        return warped_face
    return None