import cv2
import numpy as np
from PIL import Image
import os

yol = 'dataset'
tani = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_default.xml");
def getImagesAndLabels(yol):
    imagePaths = [os.path.join(yol,f) for f in os.listdir(yol)]
    faceSamples=[]
    ids = []
    labels=[]
    for imagePath in imagePaths:
        #PIL_img = Image.open(imagePath).convert('L')
        #img_numpy = np.array(PIL_img,'uint8')
        #id = int(os.path.split(imagePath)[-1].split(".")[1])
        label=(os.listdir(yol))
        print(label)
        #faces = detector.detectMultiScale(img_numpy)
        #for (x,y,w,h) in faces:
           # faceSamples.append(img_numpy[y:y+h,x:x+w])
            #ids.append(id)
            #print(label)
    #return faceSamples,ids

getImagesAndLabels(yol)
"""tani.train(faces, np.array(ids))
tani.write('trainer.yml')"""