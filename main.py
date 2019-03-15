import sys
import  numpy as np
from yolo import YOLO
from yolo_defect import YOLO_Defect
from PIL import Image
import os
import cv2
import keras
import glob
keras.backend.clear_session()
FLAGS = {}
detector = YOLO(**(FLAGS))
defection = YOLO_Defect(**(FLAGS))
path = "./test_insulator/*.jpg"
outdir = "./result"
for jpgfile in glob.glob(path):
    name = os.path.basename(jpgfile)
    print(name)
    img = Image.open(jpgfile)
    img1 = cv2.imread(jpgfile)
    result = detector.detect_image(img)
    leng=len(result)
    print(leng)
    if leng != 0:
        for i in range(leng):
            for j in range(4):
                if result[i][j] < 0:
                    result[i][j] = 0
            rect = img1[int(result[i][0]):int(result[i][2]), int(result[i][1]):int(result[i][3])]
            rect = Image.fromarray(rect)
            draw = cv2.rectangle(img1, (result[i][1], result[i][0]), (result[i][3], result[i][2]), (255, 0, 0), 2)
            quexian = defection.detect_image(rect)
            print(quexian)
            if np.any(quexian):
                quexian[0][0] += result[i][0]
                quexian[0][1] += result[i][1]
                quexian[0][2] += result[i][0]
                quexian[0][3] += result[i][1]
                draw = cv2.rectangle(img1, (quexian[0][1], quexian[0][0]), (quexian[0][3], quexian[0][2]), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(outdir, name), draw)
    else:
        cv2.imwrite(os.path.join(outdir, name), img1)



