import cv2
import numpy as np
import math
import argparse
from tracker import trackeo
import os

# class KalmanFilter:
# 	kf=cv2.KalmanFilter(4,2)
# 	kf.measurementMatrix=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
# 	kf.transitionMatrix=np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
#
# 	def predict(self):
# 		predicted = self.kf.predict()
# 		x, y= int(predicted[0]), int(predicted[1])
# 		return x, y
# 	def correct(self, coordX, coordY):
# 		measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
# 		self.kf.correct(measured)
def ann_yolo(file, w, h):
    yolo_txt = open(file)
    detecciones = []
    for data in yolo_txt:
        espacio = [0, 0, 0, 0, -1]
        pos_cadena = 0
        for cadena in range(len(data)):
            if data[cadena] == ' ':
                espacio[pos_cadena] = cadena
                pos_cadena = pos_cadena + 1
        objeto = int(data[0:int(espacio[0])])
        centrox = (float(data[int(espacio[0]):int(espacio[1])]))*1100
        centroy = (float(data[int(espacio[1]):int(espacio[2])]))*1100*h/w
        ancho = (float(data[int(espacio[2]):int(espacio[3])]))*1100
        alto = (float(data[int(espacio[3]):int(espacio[4])]))*1100*h/w
        if espacio[4] != -1:
            confi = float(data[int(espacio[4]):])
        else:
            confi = 1
        detecciones.append([objeto, [centrox, centroy, ancho, alto], confi])
    return detecciones

def visualizar(im, lista, color):
    for i in range(len(lista)):
        x1 = int(lista[i][1][0] - (lista[i][1][2] / 2))
        y1 = int(lista[i][1][1] - (lista[i][1][3] / 2))
        x2 = int(lista[i][1][0] + (lista[i][1][2] / 2))
        y2 = int(lista[i][1][1] + (lista[i][1][3] / 2))
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 1)
        cv2.putText(im, str(lista[i][0]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)



parser = argparse.ArgumentParser()
parser.add_argument('--video', default="/home/pablo.calvo/DATASET2/videos_apartamento/C027_JMG/TAB/cam2_KR1/20230424-172536-C027_JMG-TAB-0000-cam2_KR1.mp4")#required=True)
args = parser.parse_args()

video = cv2.VideoCapture(args.video)
for i in range(len(args.video)):
    if args.video[i] == '/':
        barra = i
    if args.video[i] == '.':
        punto = i
name = args.video[(barra + 1):(punto)]

#Inicialización de variables
umbral = 0.55 #Un objeto detectado es valido si la confianza es mayor del 88%
objetos = []
lista = []
trackers = []
distanciaobjeto = 0 #distancia de movimiento para considerar un objeto
numeroframe = int(0)

#Calculo de las dimensiones del vídeo
ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

yolo = "/home/pablo.calvo/Dataset/DATASET2/Objects_COCO_EYEFUL/YOLO_EYEFUL/" + "C027_JMG-TAB-2" + "/labels"

if video.isOpened() == False:
    print("Error opening video  file")

#Comienza a analizar el vídeo
while video.isOpened():
    ret, frame = video.read()
    if ret == True:
        frame = cv2.resize(frame,(1100, int(1100*alto/ancho)), interpolation = cv2.INTER_AREA) #ajustar tamaño de la imagen
        numeroframe += 1

        #Se guarda en la variable objetos las detecciones del frame actual de la forma [[classID, [bbox], confi], ...]
        try:
            file_yolo = name + "_" + str(numeroframe) + ".txt"
            file_yolo = os.path.join(yolo, file_yolo)
            objetos = ann_yolo(file_yolo, ancho, alto)
        except:
            objetos = []

        lista, trackers = trackeo(frame, objetos, lista, numeroframe, 80, trackers)

        visualizar(frame, objetos, (255, 0, 255))
        visualizar(frame, lista, (0, 255, 0))
        cv2.imshow('frame', frame)  # visualización del vídeo

    chr = cv2.waitKey(1) & 0xFF
    if chr == 27:  # Esc key to exit
        break

video.release()
cv2.destroyAllWindows()
