from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi import AutoDetectionModel
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from tqdm import tqdm
from PIL import Image
import torch

import argparse
import os
import cv2

def name_2_number(name):
    clases = ['person', 'chair', 'table', 'cutlery', 'plate', 'glass', 'shoe', 'jug', 'bowl', 'broom', 'dustbin', 'dustpan', 'napkin', 'socks', 'tablecloth', 'cupboard', 'drawer', 'bra', 'ladle', 'mop', 'mop bucket', 'panties', 'pants', 'hand', 'head', 'face', 'soup plate', 'dinner plate', 'spoon', 'knife', 'fork']
    for i in range(len(clases)):
        if name == clases[i]:
            number = int(i)
            break
    return number
parser = argparse.ArgumentParser()

parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default=1)

parser.add_argument('--sahi_method_enable', action='store_true', default=True,  # Default value: True
                    help='activates SAHI method for inference')

parser.add_argument('--folder_dir_input', type=str, help='log path',
                    default='/data1/alvaro.nieva/justoneimage')

parser.add_argument('--folder_dir_output', type=str, help='log path', default='/home/pablo.calvo/PycharmProjects/od/results')
#                    default='/data1/RESULTS-obj-det/qualitative_results/CZZZ_MMR/plates_subset_cam2_SAHI_LABELS')

parser.add_argument('--model_path', type=str, help='log path',
                    default='/data1/RESULTS-obj-det/quantitative_results_runs_yolov8/plates_subset/init_train_2_devices_bs28-ENDED/weights/best.pt')

parser.add_argument('--min_conf_threshold',  default=0.4,
                    help='if the prediction confidence is lower than this value it is not saved on the .txt file',)

args = parser.parse_args()


if __name__ == '__main__':
    files = os.listdir(args.folder_dir_input)
    files = sorted(files)
    if args.sahi_method_enable is False:
        model = YOLO(args.model_path)
        results = model.predict(files, stream=True, conf=0.5)
        # Process results generator
        for result in results:
            annotator = Annotator(files)

            boxes = result.boxes
            probs = result.probs
            im = result.plot(pil=True, probs=True)
            im.show()
            im.save("result.png")

            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                print(b)
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
    else:

        detection_model = AutoDetectionModel.from_pretrained(model_type="yolov8", model_path=args.model_path,
                                                             device="cuda:0")

        # Batch prediction is not properly working, as it returns None when it should retrun a dict with the results.
        # result = predict(
        #     model_type="yolov8",
        #     # model_path="/home/alvaro.nieva/repos/runs_yolov8/plates_subset/init_train_2_devices_bs28-ENDED/weights/best.pt",
        #     # model_path="yolov8x",
        #     model_path=args.model_path,
        #     model_device='cuda:0',#args.gpu,  # or 'cuda:0'
        #     model_confidence_threshold=0.4,
        #     source=args.folder_dir_input,
        #     return_dict=True,
        #     slice_height=480,
        #     slice_width=360,
        #     overlap_height_ratio=0.2,
        #     overlap_width_ratio=0.2,
        # )

        for file in tqdm(files, total=len(files)):
            image_path = os.path.join(args.folder_dir_input, file)
            # image_path: Location of image or numpy image matrix to slice
            result = get_sliced_prediction(
                image_path,
                detection_model,
                slice_height=480,
                slice_width=360,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            # Convert to coco to properly handling the prediction. Otherwise, you have to deal with PredictionScore
            # object with their own datatypes which are not easily converted to float.
            # result_coco = result.to_coco_annotations()
            # # result_fiftyone = result.to_fiftyone_detections()
            # # result_coco_pred = result.to_coco_predictions()
            # image_height = result_coco.image_height
            # image_width = result_coco.image_width

            bbox_list = []
            clase = detection_model.category_mapping['0']
            clase = name_2_number(clase)
            for i in range(len(result.object_prediction_list)):
                bbox = str(result.object_prediction_list[i].bbox)
                positions = []
                for j in range(len(bbox)):
                    if bbox[j] == '(':
                        positions.append(j)
                    elif bbox[j] == ',':
                        positions.append(j)
                    elif bbox[j] == ')':
                        bbox_list.append([float(bbox[positions[0]+1:positions[1]]), float(bbox[positions[1]+2:positions[2]]), float(bbox[positions[2]+2:positions[3]]), float(bbox[positions[3]+2:j]), float(result.object_prediction_list[i].score.value)])
                        break
            image_width = result.image_width
            image_height = result.image_height

            txt_path = os.path.join(args.folder_dir_output, (os.path.splitext(os.path.basename(file))[0] + '.txt'))
            txt = open(txt_path, 'w')

            for i in range(len(bbox_list)):
                clase = name_2_number(result.object_prediction_list[i].category.name)
                x = ((bbox_list[i][0] + bbox_list[i][2])/2)/image_width
                y = ((bbox_list[i][1] + bbox_list[i][3])/2)/image_height
                ancho = (bbox_list[i][2] - bbox_list[i][0])/image_width
                alto = (bbox_list[i][3] - bbox_list[i][1])/image_height
                x_centre = format(x, '.6f')
                y_centre = format(y, '.6f')
                w = format(ancho, '.6f')
                h = format(alto, '.6f')
                confi = format(bbox_list[i][4], '.6f')

                txt.write(str(clase) + " " + x_centre + " " + y_centre + " " + w + " " + h + " " + confi + os.linesep)

            txt.close()

            ##Visualizacion
            # print(file)
            # imagen = cv2.imread(os.path.join(args.folder_dir_input, file))
            # # imagen = cv2.resize(imagen, (1080, 720), interpolation=cv2.INTER_CUBIC)
            # # alto, ancho, c = imagen.shape
            # ancho = 1280
            # alto = 720
            #
            # cv2.rectangle(imagen, (int((float(x_centre) - (float(w) / 2))*ancho),
            #                        int((float(y_centre) - (float(h) / 2))*alto)), (
            #               int((float(x_centre) + (float(w) / 2))*ancho),
            #               int((float(y_centre) + (float(h) / 2))*alto)), (255, 0, 0))
            #
            # # for i in range(len(bbox_list)):
            # #     print((int(float(bbox_list[i][0])-float(bbox_list[i][2]/2))*ancho, int(float(bbox_list[i][1])-float(bbox_list[i][3]/2))*alto), (int(float(bbox_list[i][0])+float(bbox_list[i][2]/2))*ancho, int(float(bbox_list[i][1])+float(bbox_list[i][3]/2))*alto))
            # #
            # #     # cv2.rectangle(imagen, (int(float(bbox_list[i][0])-float(bbox_list[i][2]/2)), int(float(bbox_list[i][1])-float(bbox_list[i][3]/2))), (int(float(bbox_list[i][0])+float(bbox_list[i][2]/2)), int(float(bbox_list[i][1])+float(bbox_list[i][3]/2))), (255, 0, 0))
            # #     cv2.rectangle(imagen, (int(bbox_list[i][0]),
            # #                            int(bbox_list[i][1])), (
            # #                   int(bbox_list[i][2]),
            # #                   int(bbox_list[i][3])), (255, 0, 0))
            #
            # cv2.imshow("resultados", imagen)
            #
            # cv2.waitKey(10000)
            # cv2.destroyAllWindows()

        print("Finished")