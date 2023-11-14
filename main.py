import cv2
import numpy as np
import math
import argparse
from tracker import trackeo_clases, get_corners, get_iou, separar_objetos
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
        centrox = int((float(data[int(espacio[0]):int(espacio[1])]))*1100)
        centroy = int((float(data[int(espacio[1]):int(espacio[2])]))*1100*h/w)
        ancho = int((float(data[int(espacio[2]):int(espacio[3])]))*1100)
        alto = int((float(data[int(espacio[3]):int(espacio[4])]))*1100*h/w)
        if espacio[4] != -1:
            confi = float(data[int(espacio[4]):])
        else:
            confi = 1
        detecciones.append([objeto, [centrox, centroy, ancho, alto], confi])
    return detecciones

def visualizar_detecciones(im, lista, color):
    for i in range(len(lista)):
        x1 = int(lista[i][1][0] - (lista[i][1][2] / 2))
        y1 = int(lista[i][1][1] - (lista[i][1][3] / 2))
        x2 = int(lista[i][1][0] + (lista[i][1][2] / 2))
        y2 = int(lista[i][1][1] + (lista[i][1][3] / 2))
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 1)
        cv2.putText(im, str(lista[i][0]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def visualizar_objeto(im, lista, color, clase):
    for i in range(len(lista[clase])):
        x1 = int(lista[clase][i][1][0] - (lista[clase][i][1][2] / 2))
        y1 = int(lista[clase][i][1][1] - (lista[clase][i][1][3] / 2))
        x2 = int(lista[clase][i][1][0] + (lista[clase][i][1][2] / 2))
        y2 = int(lista[clase][i][1][1] + (lista[clase][i][1][3] / 2))
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 1)
        cv2.putText(im, str(i), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def filtrar_objetos(lista, thr_iou, thr_det, limites):
    borrar = []
    for i in range(len(lista)):
        j = 0
        while j < len(lista):
            if lista[i][0] == lista[j][0]:
                iou = get_iou(get_corners(lista[i]), get_corners(lista[j]))
                if i != j and iou > thr_iou:
                    borrar.append(j)
            j += 1
    i = len(lista)
    while i > -1:
        for j in range(len(borrar)):
            if i == borrar[j]:
                lista.pop(i)
                break
        i -= 1

    i = 0
    while i < len(lista):
        x1, y1, x2, y2 = get_corners(lista[i])
        clase = lista[i][0]
        if lista[i][2] < thr_det or x1 < limites[clase][0] or y1 < limites[clase][1] or x2 > limites[clase][2] or y2 > limites[clase][3] or lista[i][1][2] > limites[clase][4] or lista[i][1][3] > limites[clase][5]:
            lista.pop(i)
        else:
            i += 1

    return lista


parser = argparse.ArgumentParser()
parser.add_argument('--video', default="/home/pablo.calvo/DATASET2/videos_apartamento/C027_JMG/TAB/cam2_KR1/20230424-172536-C027_JMG-TAB-0000-cam2_KR1.mp4")#required=True)
parser.add_argument('--num_ob', default=31)
args = parser.parse_args()

args.video = "/home/pablo.calvo/Videos2/alvaro.nieva/DISTRACTORS/20230714-135720-CZZZ_MMR-TAB-0000-cam2_KR1.mp4"
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
for i in range(args.num_ob):
    lista.append([])
    trackers.append([])

distanciaobjeto = 0 #distancia de movimiento para considerar un objeto
numeroframe = int(0)

#Calculo de las dimensiones del vídeo
ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

#eyeful contiene el numero de objetos máximos que deberían de aparecer de cada clase
# eyeful = [1, 4, 1, 15, 5, 5, 2, 1, 1, 1, 1, 1, 5, 2, 1, ??(cupboard), ??(drawer), ??(bra), 1, 1, 1, ??(panties), ??(pants), 2, 1, 1, 5, 5, 5, 5, 5]
eyeful = [1, 4, 1, 15, 5, 5, 2, 1, 1, 1, 1, 1, 5, 2, 1, 0, 0, 0, 1, 1, 1, 0, 0, 2, 1, 1, 5, 5, 5, 5, 5]
trackers_class = (3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
limites_objetos = []
for i in range(args.num_ob):
    limites_objetos.append([0, 0, 1100, int(1100*alto/ancho), 1100, 1100])

limites_objetos[0] = [350, 200, 1100, 550, 100, 100] #Es de la clase 4 (plato)

# yolo = "/home/pablo.calvo/Dataset/DATASET2/Objects_COCO_EYEFUL/YOLO_EYEFUL/" + "C027_JMG-TAB-2" + "/labels"
yolo = "/home/pablo.calvo/Videos2/RESULTS-obj-det/qualitative_results/CZZZ_MMR/plates_subset_cam2_LABELS/labels"

if video.isOpened() == False:
    print("Error opening video  file")

#Comienza a analizar el vídeo
while video.isOpened():
    ret, frame = video.read()
    if ret == True:
        frame = cv2.resize(frame,(1100, int(1100*alto/ancho)), interpolation=cv2.INTER_AREA) #ajustar tamaño de la imagen
        numeroframe += 1

        #Se guarda en la variable objetos las detecciones del frame actual de la forma [[classID, [bbox], confi], ...]
        try:
            file_yolo = name + "_" + str(numeroframe) + ".txt"
            file_yolo = "%#06d.txt" % (numeroframe)
            file_yolo = os.path.join(yolo, file_yolo)
            objetos = ann_yolo(file_yolo, ancho, alto)
        except:
            objetos = []
        objetos = filtrar_objetos(objetos, 0.85, 0.8, limites_objetos)

        if numeroframe == 1:
            print("CUIDADO, quitar líneas 141 y 142")
        for i in range(len(objetos)):
            objetos[i][0] = 4

        lista, trackers = separar_objetos(frame, objetos, lista, numeroframe, 40, trackers, eyeful, trackers_class)
        lista, trackers = trackeo_clases(frame, lista, numeroframe, trackers, trackers_class)

        # visualizar_detecciones(frame, objetos, (255, 0, 255))
        visualizar_objeto(frame, lista, (0, 255, 0), 0)
        visualizar_objeto(frame, lista, (0, 255, 0), 4)
        cv2.imshow('frame', frame)  # visualización del vídeo

    chr = cv2.waitKey(5) & 0xFF
    if chr == 27:  # Esc key to exit
        break

video.release()
cv2.destroyAllWindows()
