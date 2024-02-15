# Tracker_od
## Requirements
- python 3
- os
- opencv
- json
- argparse
- dlib
- pandas


## main.py
`main.py` es un script para el tracking y la asociación de diferentes objetos. Para su funcionamiento se comienza introduciendo el nombre del vídeo y el archivo de lectura de datos:

Command: `main.py --video --json [--num_ob]`
- --video. Is the input video file path.
- --json. Is the annotation json file path
- --num_ob. Is the objects to analyze. Default: 31.

En su funcionamiento, se procede a la lectura del vídeo frame a frame, realizando la lectura de las detecciones de objetos en cada frame.
Con todos ellos se realiza un filtrado en función de la confianza,  la posición y el tamaño de cada uno de ellos, pudiendo modificar sus valores en la variable "limites_objetos", 
que presenta el siguiente formato para cada una de las clases: [x min, y min, x max, y max, ancho max, alto max].

Con una variable (objetos) que contiene las detecciones válidas, se procede a la asociación con los objetos anteriores (implementado actualmente con la distancia entre los centros),
asumiendo que se trata del mismo objetos para la detección de la misma clase con la distancia mínima por debajo del umbral establecido.

También se establece un límite de objetos por cada clase, borrando la detección más antigua si se supera el número de objetos almacenados.

Finalmente, se realiza una predicción de los objetos almacenados que no hayan sido asociados (que no se detectan) en el frame actual.
