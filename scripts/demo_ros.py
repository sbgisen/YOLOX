#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch
import rospkg
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        legacy=False,
    ):
        model_name = rospy.get_param('~model_name', 'yolox-s')
        exp = get_exp(None, model_name)

        exp.test_conf = 0.5
        exp.nmsthre = 0.3
        exp.test_size = (640, 640)

        self.model = exp.get_model()

        self.device = rospy.get_param('~device', 'gpu')
        if self.device == "gpu":
            self.model.cuda()
        self.model.eval()

        pkg = rospkg.RosPack().get_path('yolox')
        ckpt_file = rospy.get_param('~ckpt', pkg+'/models/yolox_s.pth')
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        self.model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

        self.cls_names = COCO_CLASSES
        self.decoder = None
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.test_size = exp.test_size
        self.fp16 = False
        self.nmsthre = exp.nmsthre
        self.preproc = ValTransform(legacy=legacy)

        self._bridge = CvBridge()
        self._pub = rospy.Publisher('~output', Image, queue_size=1)
        rospy.Subscriber('~image', Image, self.inference, queue_size=1)

    def inference(self, msg):

        img = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        img_info = {"id": 0}
        img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        result_image = self.visual(outputs[0], img_info, self.confthre)

        image_msg = self._bridge.cv2_to_imgmsg(result_image, 'bgr8')
        self._pub.publish(image_msg)

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


if __name__ == "__main__":
    rospy.init_node('yolox')
    _ = Predictor()
    rospy.spin()
