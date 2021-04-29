import colorsys
import os
import pickle

import cv2
import keras
import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from nets.facenet import facenet
from nets_retinaface.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import (Alignment_1, BBoxUtility, compare_faces,
                         letterbox_image, retinaface_correct_boxes)


def cv2ImgAddText(img, label, left, top, textColor=(255, 255, 255)):
    img = Image.fromarray(np.uint8(img))
    # 设置字体
    font = ImageFont.truetype(font='font/simhei.ttf', size=20)

    draw = ImageDraw.Draw(img)
    label = label.encode('utf-8')
    draw.text((left, top), str(label,'UTF-8'), fill=textColor, font=font)
    return np.asarray(img)
    
#--------------------------------------#
#   一定注意backbone和model_path的对应。
#   在更换facenet_model后，
#   一定要注意重新编码人脸。
#--------------------------------------#
class Retinaface(object):
    _defaults = {
        "retinaface_model_path" : 'model_data/retinaface_mobilenet025.h5',
        #-----------------------------------#
        #   可选retinaface_backbone有
        #   mobilenet和resnet50
        #-----------------------------------#
        "retinaface_backbone"   : "mobilenet",
        "confidence"            : 0.5,
        "iou"                   : 0.3,
        #----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #   输入图像大小会大幅度地影响FPS，想加快检测速度可以减少input_shape。
        #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        #   keras代码中主干为mobilenet时存在小bug，当输入图像的宽高不为32的倍数
        #   会导致检测结果偏差，主干为resnet50不存在此问题。
        #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        #----------------------------------------------------------------------#
        "retinaface_input_shape": [640, 640, 3],
        "letterbox_image"       : True,

        "facenet_model_path"    : 'model_data/facenet_mobilenet.h5',
        #-----------------------------------#
        #   可选facenet_backbone有
        #   mobilenet和inception_resnetv1
        #-----------------------------------#
        "facenet_backbone"      : "inception_resnetv1",
        "facenet_input_shape"   : [160,160,3],
        "facenet_threhold"      : 0.9,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Retinaface+facenet
    #---------------------------------------------------#
    def __init__(self, encoding=0, **kwargs):
        self.__dict__.update(self._defaults)
        if self.retinaface_backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50
        self.bbox_util = BBoxUtility(nms_thresh=self.iou)
        self.generate()
        self.anchors = Anchors(self.cfg, image_size=(self.retinaface_input_shape[0], self.retinaface_input_shape[1])).get_anchors()

        try:
            self.known_face_encodings = np.load("model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone))
            self.known_face_names     = np.load("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone))
        except:
            if not encoding:
                print("载入已有人脸特征失败，请检查model_data下面是否生成了相关的人脸特征文件。")
            pass

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        self.retinaface = RetinaFace(self.cfg, self.retinaface_backbone)
        self.facenet    = facenet(self.facenet_input_shape, backbone=self.facenet_backbone, mode='predict')
        
        print('Loading weights into state dict...')
        self.retinaface.load_weights(self.retinaface_model_path, by_name=True)
        self.facenet.load_weights(self.facenet_model_path, by_name=True)
        print('Finished!')

    def encode_face_dataset(self, image_paths, names):
        face_encodings = []

        for index, path in enumerate(tqdm(image_paths)):
            image = Image.open(path)
            image = np.array(image, np.float32)
            old_image = image.copy()
            
            im_height, im_width, _ = np.shape(image)

            scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
            scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                                np.shape(image)[1], np.shape(image)[0]]

            if self.letterbox_image:
                image = letterbox_image(image,[self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
                anchors = self.anchors
            else:
                anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

            #---------------------------------------------------#
            #   图片预处理，归一化
            #---------------------------------------------------#
            photo = np.expand_dims(preprocess_input(image),0)

            #---------------------------------------------------#
            #   将处理完的图片传入Retinaface网络当中进行预测
            #---------------------------------------------------#
            preds = self.retinaface.predict(photo)

            #---------------------------------------------------#
            #   Retinaface网络的解码，最终我们会获得预测框
            #   将预测结果进行解码和非极大抑制
            #---------------------------------------------------#
            results = self.bbox_util.detection_out(preds,anchors,confidence_threshold=self.confidence)

            if len(results)<=0:
                print(names[index], "：未检测到人脸")
                continue

            results = np.array(results)
            if self.letterbox_image:
                results = retinaface_correct_boxes(results, np.array((self.retinaface_input_shape[0], self.retinaface_input_shape[1])), np.array([im_height, im_width]))
            
            results[:,:4] = results[:,:4]*scale
            results[:,5:] = results[:,5:]*scale_for_landmarks

            #---------------------------------------------------#
            #   选取最大的人脸框。
            #---------------------------------------------------#
            best_face_location = None
            biggest_area = 0
            for result in results:
                left, top, right, bottom = result[0:4]

                w = right - left
                h = bottom - top
                if w*h > biggest_area:
                    biggest_area = w*h
                    best_face_location = result

            #---------------------------------------------------#
            #   截取图像
            #---------------------------------------------------#
            crop_img = old_image[int(best_face_location[1]):int(best_face_location[3]), int(best_face_location[0]):int(best_face_location[2])]
            landmark = np.reshape(best_face_location[5:],(5,2)) - np.array([int(best_face_location[0]),int(best_face_location[1])])
            crop_img,_ = Alignment_1(crop_img,landmark)

            crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
            crop_img = np.expand_dims(crop_img,0)
            #---------------------------------------------------#
            #   利用图像算取长度为128的特征向量
            #---------------------------------------------------#
            face_encoding = self.facenet.predict(crop_img)[0]
            face_encodings.append(face_encoding)

        np.save("model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone),face_encodings)
        np.save("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone),names)

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image = np.array(image, np.float32)
        old_image = np.array(image.copy(), np.uint8)
        #---------------------------------------------------#
        #   Retinaface检测部分-开始
        #---------------------------------------------------#
        # 数据的预处理
        im_height, im_width, _ = np.shape(image)

        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0]]

        if self.letterbox_image:
            image = letterbox_image(image,[self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        #---------------------------------------------------#
        #   图片预处理，归一化
        #---------------------------------------------------#
        photo = np.expand_dims(preprocess_input(image),0)

        #---------------------------------------------------#
        #   将处理完的图片传入Retinaface网络当中进行预测
        #---------------------------------------------------#
        preds = self.retinaface.predict(photo)

        #---------------------------------------------------#
        #   Retinaface网络的解码，最终我们会获得预测框
        #   将预测结果进行解码和非极大抑制
        #---------------------------------------------------#
        results = self.bbox_util.detection_out(preds,anchors,confidence_threshold=self.confidence)

        #---------------------------------------------------#
        #   如果没有预测框则返回原图
        #---------------------------------------------------#
        if len(results)<=0:
            return old_image

        results = np.array(results)
        if self.letterbox_image:
            results = retinaface_correct_boxes(results, np.array((self.retinaface_input_shape[0], self.retinaface_input_shape[1])), np.array([im_height, im_width]))
        
        #---------------------------------------------------#
        #   4人脸框置信度
        #   :4是框的坐标
        #   5:是人脸关键点的坐标
        #---------------------------------------------------#
        results[:,:4] = results[:,:4]*scale
        results[:,5:] = results[:,5:]*scale_for_landmarks
        #---------------------------------------------------#
        #   Retinaface检测部分-结束
        #---------------------------------------------------#

        #-----------------------------------------------#
        #   Facenet编码部分-开始
        #-----------------------------------------------#
        face_encodings = []
        for result in results:
            #----------------------#
            #   图像截取，人脸矫正
            #----------------------#
            crop_img = np.array(old_image)[int(result[1]):int(result[3]), int(result[0]):int(result[2])]
            landmark = np.reshape(result[5:],(5,2)) - np.array([int(result[0]),int(result[1])])
            crop_img,_ = Alignment_1(crop_img,landmark)

            #----------------------#
            #   人脸编码
            #----------------------#
            # 不失真的resize，然后进行归一化
            crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
            crop_img = np.expand_dims(crop_img,0)
            # 利用图像算取长度为128的特征向量
            face_encoding = self.facenet.predict(crop_img)[0]
            face_encodings.append(face_encoding)
        #-----------------------------------------------#
        #   Facenet编码部分-结束
        #-----------------------------------------------#

        #-----------------------------------------------#
        #   人脸特征比对-开始
        #-----------------------------------------------#
        face_names = []
        for face_encoding in face_encodings:
            # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
            matches, face_distances = compare_faces(self.known_face_encodings, face_encoding, tolerance = self.facenet_threhold)
            name = "Unknown"
            # 找到已知最贴近当前人脸的人脸序号
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)
        #-----------------------------------------------#
        #   人脸特征比对-结束
        #-----------------------------------------------#
        
        for i, b in enumerate(results):
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
            
            name = face_names[i]
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(old_image, name, (b[0] , b[3] - 15), font, 0.75, (255, 255, 255), 2) 
            #--------------------------------------------------------------#
            #   cv2不能写中文，加上这段可以，但是检测速度会有一定的下降。
            #   如果不是必须，可以换成cv2只显示英文。
            #--------------------------------------------------------------#
            old_image = cv2ImgAddText(old_image, name, b[0]+5 , b[3] - 25)
        return old_image

