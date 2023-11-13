import cv2
import numpy as np
import math
import argparse

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


parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True)
args = parser.parse_args()

video=cv2.VideoCapture(args.video)

#Inicialización de variables
umbral = 0.55 #Un objeto detectado es valido si la confianza es mayor del 88%
l = 0
primerobjeto = 0
sospechosos = []
abandonados = []
k = 0
distanciaobjeto = 0 #distancia de movimiento para considerar un objeto
numeroframe = 0

#Calculo de las dimensiones del vídeo
ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


if video.isOpened() == False:
    print("Error opening video  file")

#Comienza a analizar el vídeo
while video.isOpened():
    ret, frame = video.read()
    if ret == True:
        frame=cv2.resize(frame,(1100, int(1100*alto/ancho)), interpolation = cv2.INTER_AREA) #ajustar tamaño de la imagen
        cv2.imshow('frame', frame) #visualización del vídeo original
        numeroframe += 1


    chr = cv2.waitKey(1) & 0xFF
    if chr == 27:  # Esc key to exit
        break

video.release()
cv2.destroyAllWindows()
