import cv2
from matplotlib import pyplot as plt
import numpy as np

filtro = 0

#caminho onde estão os pesos
def detecta_face(img):
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    return faces

def setBlur(img,faces):
    img_copy = img.copy()
    if(len(faces)>0):
        for (x,y,w,h) in faces:
            img_copy[y:y+h, x:x+w]=cv2.medianBlur(img_copy[y:y+h, x:x+w],35)
    return img_copy

def setBlurBack(img,faces):
    img_copy = img.copy()
    if(len(faces)>0):
        for (x,y,w,h) in faces:
            face = img_copy[y:y+h, x:x+w]
            img_copy=cv2.medianBlur(img_copy,35)
            img_copy[y:y+h, x:x+w] = face
    return img_copy

def setEdge(img,faces):
    img_copy = img.copy()
    img_grey = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    if(len(faces)>0):
        for (x,y,w,h) in faces:
            img_grey[y:y+h, x:x+w]=cv2.Canny(img_grey[y:y+h, x:x+w],30,50)
    return img_grey

def mouse_click(event, x, y, flags, param):
    global filtro
    # se foi click do botao direito 
    if event == cv2.EVENT_RBUTTONDOWN:
        filtro += 1
        
    if event == cv2.EVENT_LBUTTONDOWN:
        filtro = 0 
        

cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    faces = detecta_face(img)
    img_blur = setBlur(img,faces)
    back_blur = setBlurBack(img,faces)
    img_edge = setEdge(img,faces)

    cv2.imshow('img',img)
    if(filtro==0):
        cv2.imshow('img_blur', img_blur)
    elif(filtro==1):
        cv2.imshow('back_blur', back_blur)
    elif(filtro==2):
        cv2.imshow('img_edge', img_edge)
    
    cv2.setMouseCallback('img', mouse_click)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break


# fecha a janela.
cv2.destroyAllWindows()
cap.release()