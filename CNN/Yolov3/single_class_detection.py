import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys

class SINGLE:
    def __init__(self, path):
        try:
            self.threshold = 0.5
            self.image_size = 320

            self.image = cv2.imread(path)
            # print(image.shape)
            self.original_height, self.original_width = self.image.shape[0], self.image.shape[1]


            self.v3_neural_network = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
            self.class_names = []
            with open('coco.names', 'r') as f:
                for i in f.readlines():
                    self.class_names.append(i.strip())
            self.blob = cv2.dnn.blobFromImage(self.image, 1/255, (320, 320), True, crop=False)
            # print(blob.shape)
            self.v3_neural_network.setInput(self.blob)
            self.complete_layers = self.v3_neural_network.getLayerNames()
            # print(complete_layers)
            self.output_layers_index = self.v3_neural_network.getUnconnectedOutLayers()
            # print(output_layers_index)
            self.output_layers = [self.complete_layers[i-1] for i in self.output_layers_index]
            # print(output_layers)
            self.output_data = self.v3_neural_network.forward(self.output_layers)
            # print(output_data)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type}\n Error Msg: {er_msg}\n Error Lineno: {er_line.tb_lineno}')


    def bounding_boxes(self):
        try:
            self.class_ids = []
            self.confidence_scores = []
            self.cordinates = []

            for i in self.output_data:
                for j in i:
                    prob_values = j[5:]
                    max_class_index = np.argmax(prob_values)
                    max_class_value = prob_values[max_class_index]

                    if max_class_value > self.threshold:
                        w, h = int(j[2] * self.image_size), int(j[3] * self.image_size)
                        x, y = int(j[0] * self.image_size - w / 2), int(j[1] * self.image_size - h / 2)
                        self.cordinates.append([x,y,w,h])
                        self.confidence_scores.append(max_class_value)
                        self.class_ids.append(max_class_index)
            self.final_box = cv2.dnn.NMSBoxes(self.cordinates, self.confidence_scores, self.threshold, 0.6)
            return self.final_box, self.class_ids, self.confidence_scores, self.cordinates
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type}\n Error Msg: {er_msg}\n Error Lineno: {er_line.tb_lineno}')

    def predictions(self, final_box, class_ids, confidence_scores, cordinates):
        try:
            width_ratio = self.original_width / self.image_size
            height_ratio = self.original_height / self.image_size
            for i in self.final_box:
                x, y, w, h = self.cordinates[i]
                x = int(x * width_ratio)
                y = int(y * height_ratio)
                w = int(w * width_ratio)
                h = int(h * height_ratio)
                cnf = str(self.confidence_scores[i])
                text = str(self.class_names[self.class_ids[i]]) + '---' + cnf
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.rectangle(self.image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(self.image, text, (x, y-2), font, 1, (0,0,255),1, cv2.LINE_AA)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type}\n Error Msg: {er_msg}\n Error Lineno: {er_line.tb_lineno}')


if __name__ == "__main__":
    try:
        single = SINGLE('car_image.jpg')
        final_box, class_ids, confidence_scores, cordinates = single.bounding_boxes()
        single.predictions(final_box, class_ids, confidence_scores, cordinates)
        # print(final_box)
        # print(cordinates)
        # print(confidence_scores)
        # print(class_ids)

        cv2.imshow('car', single.image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f'Error Type: {er_type}\n Error Msg: {er_msg}\n Error Lineno: {er_line.tb_lineno}')
