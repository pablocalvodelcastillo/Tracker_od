import math
import dlib
import cv2


class Tracker():
    def __init__(self):
        self.t = dlib.correlation_tracker()

    def start(self, im, px1, py1, px2, py2):
        # p0 is leftupper position of the obj, p1 is rightbottom of the obj
        self.t.start_track(im, dlib.rectangle(px1, py1, px2, py2))

    def track(self, im):
        self.t.update(im)
        # print self.t.get_position()
        position = self.t.get_position()
        return int(position.left()), int(position.top()), int(position.right()), int(position.bottom())


def trackeo(im, objetos, lista, numeroframe, dist_min, trackers):
    for i in range(len(lista)):
        px1, py1, px2, py2 = trackers[i].track(im)
        lista[i][1] = [int((px1+px2)/2), int((py1+py2)/2), int(px2-px1), int(py2-py1)]

    for i in range(len(objetos)):
        x1 = int(objetos[i][1][0] - (objetos[i][1][2] / 2))
        y1 = int(objetos[i][1][1] - (objetos[i][1][3] / 2))
        x2 = int(objetos[i][1][0] + (objetos[i][1][2] / 2))
        y2 = int(objetos[i][1][1] + (objetos[i][1][3] / 2))
        distanciamin = dist_min
        for j in range(len(lista)):
            if objetos[i][0] == lista[j][0]:
                distx = abs(lista[j][1][0] - objetos[i][1][0])
                disty = abs(lista[j][1][1] - objetos[i][1][1])
                distancia = math.sqrt(pow(distx, 2) + pow(disty, 2))
                if distancia < distanciamin:
                    distanciamin = distancia
                    indice_objetos = j
        if distanciamin < dist_min:  # se corrige con la medida más cercana
            lista[indice_objetos][1] = objetos[i][1]
            lista[indice_objetos][3] = numeroframe
            trackers[indice_objetos].start(im, x1, y1, x2, y2)
        else:  # se crea un nuevo objeto ya que no se puede asociar, y se crea el tracker
            lista.append([objetos[i][0], objetos[i][1], numeroframe, numeroframe])
            tracker = Tracker()
            tracker.start(im, x1, y1, x2, y2)
            trackers.append(tracker)

    while i < len(lista):
        # if (predicciones_objetos[i][2]<numeroframe) and ((numeroframe-predicciones_objetos[i][1])>15): #si no se actualiza y se han introducido más de 15 medidas se almacena la predicción
        # 	predicciones_objetos[i][0]=kalman_objetos[i].predict
        if (numeroframe - lista[i][3]) > 80:  # se borra el filtro, se desplazan las variables y se crean posiciones libres al final
            lista.pop(i)
        else:
            i += 1

        # posible_abandono.pop(i)
        # posible_abandono.append([0,0,(0,0)])
        # predicciones_objetos.pop(i)
        # kalman_objetos.pop(i)
        # kalman_objetos.append(KalmanFilter())
        # predicciones_objetos.append([(0,0),0,0])
        # numob-=1

    return lista, trackers
