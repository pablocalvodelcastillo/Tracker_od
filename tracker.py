import math
import dlib
import cv2
import numpy as np


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

def get_iou(pred_bbox, gt_bbox):
    '''
    :param pred_bbox: [x1, y1, x2, y2]
    :param gt_bbox:  [x1, y1, x2, y2]
    :return: iou
    '''

    ixmin = max(pred_bbox[0], gt_bbox[0])
    iymin = max(pred_bbox[1], gt_bbox[1])
    ixmax = min(pred_bbox[2], gt_bbox[2])
    iymax = min(pred_bbox[3], gt_bbox[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.)
    ih = np.maximum(iymax - iymin + 1.0, 0.)

    inters = iw * ih
    # uni=s1+s2-inters
    uni = (pred_bbox[2] - pred_bbox[0] + 1.0) * (pred_bbox[3] - pred_bbox[1] + 1.0) + \
          (gt_bbox[2] - gt_bbox[0] + 1.0) * (gt_bbox[3] - gt_bbox[1] + 1.0) - inters
    # uni1 = (pred_bbox[2] - pred_bbox[0] + 1.0) * (pred_bbox[3] - pred_bbox[1] + 1.0)
    uni2 = (gt_bbox[2] - gt_bbox[0] + 1.0) * (gt_bbox[3] - gt_bbox[1] + 1.0)
    # uni = min(uni1, uni2)

    iou = inters / uni2

    return iou

def get_corners(lista):
    x1 = int(lista[1][0] - (lista[1][2] / 2))
    y1 = int(lista[1][1] - (lista[1][3] / 2))
    x2 = int(lista[1][0] + (lista[1][2] / 2))
    y2 = int(lista[1][1] + (lista[1][3] / 2))

    return x1, y1, x2, y2

def separar_objetos(im, objetos, lista, numeroframe, dist_min, trackers, eyeful, trackers_class):
    for i in range(len(objetos)):
        clase = objetos[i][0]
        distanciamin = dist_min
        x1, y1, x2, y2 = get_corners(objetos[i])
        for j in range(len(lista[clase])):
            distx = abs(lista[clase][j][1][0] - objetos[i][1][0])
            disty = abs(lista[clase][j][1][1] - objetos[i][1][1])
            distancia = math.sqrt(pow(distx, 2) + pow(disty, 2))
            if distancia < distanciamin:
                distanciamin = distancia
                indice_objetos = j
        if distanciamin < dist_min:  # se guarda la medida más cercana, se corrige el tracker
            lista[clase][indice_objetos][1] = objetos[i][1]
            lista[clase][indice_objetos][3] = numeroframe
            if clase in trackers_class:
                trackers[clase][indice_objetos].start(im, x1, y1, x2, y2)
        else:  # se crea un nuevo objeto ya que no se puede asociar, y se crea el tracker
            if len(lista[clase]) <= eyeful[clase]:
                lista[clase].append([objetos[i][0], objetos[i][1], numeroframe, numeroframe])
                if clase in trackers_class:
                    tracker = Tracker()
                    tracker.start(im, x1, y1, x2, y2)
                    trackers[clase].append(tracker)
            else:
                antiguo = numeroframe + 5
                borrar = -1
                for i in range(len(lista[clase])):
                    if lista[clase][i][3] < antiguo:
                        antiguo = lista[clase][i][3]
                        borrar = i
                lista[clase][borrar][1] = objetos[i][1]
                lista[clase][borrar][2] = numeroframe
                lista[clase][borrar][3] = numeroframe
    return lista, trackers



def trackeo_clases(im, lista, numeroframe, trackers, trackers_class):
    for clase in range(len(lista)):
        if clase in trackers_class:
            for i in range(len(lista[clase])):
                if lista[clase][i][3] < numeroframe-2:
                    px1, py1, px2, py2 = trackers[clase][i].track(im)
                    lista[clase][i][1] = [int((px1+px2)/2), int((py1+py2)/2), int(px2-px1), int(py2-py1)]
        i = 0
        while i < len(lista[clase]):
            if (numeroframe - lista[clase][i][3]) > 200:  # se borra el filtro, se desplazan las variables y se crean posiciones libres al final
                lista[clase].pop(i)
                trackers[clase].pop(i)
            else:
                i += 1

    return lista, trackers

# def trackeo(im, objetos, lista, numeroframe, dist_min, trackers):
#     for i in range(len(lista)):
#         if lista[i][3] < numeroframe-1:
#             px1, py1, px2, py2 = trackers[i].track(im)
#             lista[i][1] = [int((px1+px2)/2), int((py1+py2)/2), int(px2-px1), int(py2-py1)]
#         # j = 0
#         # while j < len(lista):
#         #     iou = get_iou(get_corners(lista[i]), get_corners(lista[j]))
#         #     if i != j and iou > 0.8:
#         #         lista.pop(j)
#         #         trackers.pop(j)
#         #     else:
#         #         j += 1
#
#
#     for i in range(len(objetos)):
#         x1, y1, x2, y2 = get_corners(objetos[i])
#         distanciamin = dist_min
#         for j in range(len(lista)):
#             if int(objetos[i][0]) == int(lista[j][0]):
#                 distx = abs(lista[j][1][0] - objetos[i][1][0])
#                 disty = abs(lista[j][1][1] - objetos[i][1][1])
#                 distancia = math.sqrt(pow(distx, 2) + pow(disty, 2))
#                 if distancia < distanciamin:
#                     distanciamin = distancia
#                     indice_objetos = j
#         if distanciamin < dist_min:  # se corrige con la medida más cercana
#             lista[indice_objetos][1] = objetos[i][1]
#             lista[indice_objetos][3] = numeroframe
#             trackers[indice_objetos].start(im, x1, y1, x2, y2)
#         else:  # se crea un nuevo objeto ya que no se puede asociar, y se crea el tracker
#             lista.append([objetos[i][0], objetos[i][1], numeroframe, numeroframe])
#             tracker = Tracker()
#             tracker.start(im, x1, y1, x2, y2)
#             trackers.append(tracker)
#     i = 0
#     while i < len(lista):
#         # if (predicciones_objetos[i][2]<numeroframe) and ((numeroframe-predicciones_objetos[i][1])>15): #si no se actualiza y se han introducido más de 15 medidas se almacena la predicción
#         # 	predicciones_objetos[i][0]=kalman_objetos[i].predict
#         if (numeroframe - lista[i][3]) > 80:  # se borra el filtro, se desplazan las variables y se crean posiciones libres al final
#             lista.pop(i)
#             trackers.pop(i)
#         else:
#             i += 1
#
#         # posible_abandono.pop(i)
#         # posible_abandono.append([0,0,(0,0)])
#         # predicciones_objetos.pop(i)
#         # kalman_objetos.pop(i)
#         # kalman_objetos.append(KalmanFilter())
#         # predicciones_objetos.append([(0,0),0,0])
#         # numob-=1
#
#     return lista, trackers
