import colorsys
import os
import pickle
import time

import cv2
import keras
import numpy as np
import tqdm
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from nets.facenet import facenet
from nets_retinaface.retinaface import RetinaFace
from retinaface import Retinaface
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import (Alignment_1, BBoxUtility, compare_faces,
                         letterbox_image, retinaface_correct_boxes)


class FPS_Retinaface(Retinaface):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def get_FPS(self, image, test_interval):
        image = np.array(image, np.float32)
        old_image = np.array(image.copy(), np.uint8)
        im_height, im_width, _ = np.shape(image)

        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0]]
        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        photo = np.expand_dims(preprocess_input(image),0)
        preds = self.retinaface.predict(photo)
        results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

        if len(results)>0:
            results = np.array(results)
            if self.letterbox_image:
                results = retinaface_correct_boxes(results, np.array([self.retinaface_input_shape[0], self.retinaface_input_shape[1]]), np.array([im_height, im_width]))
        
            results[:,:4] = results[:,:4]*scale
            results[:,5:] = results[:,5:]*scale_for_landmarks
            
            face_encodings = []
            for result in results:
                crop_img = np.array(old_image)[int(result[1]):int(result[3]), int(result[0]):int(result[2])]
                landmark = np.reshape(result[5:],(5,2)) - np.array([int(result[0]),int(result[1])])
                crop_img,_ = Alignment_1(crop_img,landmark)

                crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
                crop_img = np.expand_dims(crop_img,0)
                face_encoding = self.facenet.predict(crop_img)[0]
                face_encodings.append(face_encoding)

            face_names = []
            for face_encoding in face_encodings:
                matches, face_distances = compare_faces(self.known_face_encodings, face_encoding, tolerance = self.facenet_threhold)
                name = "Unknown"
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                face_names.append(name)

        t1 = time.time()
        for _ in range(test_interval):
            preds = self.retinaface.predict(photo)
            results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

            if len(results)>0:
                results = np.array(results)
                if self.letterbox_image:
                    results = retinaface_correct_boxes(results, np.array([self.retinaface_input_shape[0], self.retinaface_input_shape[1]]), np.array([im_height, im_width]))
                
                results[:,:4] = results[:,:4]*scale
                results[:,5:] = results[:,5:]*scale_for_landmarks

                face_encodings = []
                for result in results:
                    crop_img = np.array(old_image)[int(result[1]):int(result[3]), int(result[0]):int(result[2])]
                    landmark = np.reshape(result[5:],(5,2)) - np.array([int(result[0]),int(result[1])])
                    crop_img,_ = Alignment_1(crop_img,landmark)

                    crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
                    crop_img = np.expand_dims(crop_img,0)
                    face_encoding = self.facenet.predict(crop_img)[0]
                    face_encodings.append(face_encoding)

                face_names = []
                for face_encoding in face_encodings:
                    matches, face_distances = compare_faces(self.known_face_encodings, face_encoding, tolerance = self.facenet_threhold)
                    name = "Unknown"
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                    face_names.append(name)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

if __name__ == '__main__':
    retinaface = FPS_Retinaface()
    test_interval = 100
    img = Image.open('img/obama.jpg')
    tact_time = retinaface.get_FPS(img, test_interval)
    print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
