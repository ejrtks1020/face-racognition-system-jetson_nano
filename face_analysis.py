# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 


from __future__ import division

import glob
import os.path as osp

import numpy as np
import onnxruntime
from numpy.linalg import norm

from ..model_zoo import model_zoo
from ..utils import DEFAULT_MP_NAME, ensure_available
from .common import Face
import time
__all__ = ['FaceAnalysis']

class FaceAnalysis:
    def __init__(self, name=DEFAULT_MP_NAME, root='~/.insightface', allowed_modules=None, **kwargs):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        self.model_dir = ensure_available('models', name, root=root)
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def iou(self, box1, box2):
        # box = (x1, y1, x2, y2)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (box1_area + box2_area - inter)
        return iou
        
    def get(self, img, max_num=0):
        s = time.time()
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        e = time.time()
        #print('bboxes', bboxes, type(bboxes))
        #print('kpss', kpss, type(kpss))
        print('detection time: ', e - s)

        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            #print('face : ', face, '\n')
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                #print('model : ', model , '\n')
                model.get(img, face)
            ret.append(face)
            #print('ret : ', ret, '\n')
        return ret

    def new_get(self, img, prev_result, max_num=0):
        s = time.time()
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        e = time.time()
        #print('bboxes', bboxes)
        #print('kpss', kpss)
        #print('detection time: ', e - s)

        if bboxes.shape[0] == 0:
            return []

        overlapped_bboxes = []
        non_overlapped_bboxes = []
        new_kpss = []
        if prev_result:
            for index, i in enumerate(bboxes):
                overlap = False
                for j in prev_result:
                    iou_score = self.iou(i, j['bbox'])
                    #print('iou score : ', iou_score)
                    if iou_score > 0.5:
                        # 만약 bbox가 겹치면 기존의 return dict에서 새로운 bbox로만 바꿈
                        j['bbox'] = np.array(i[0:4], dtype = np.float32)
                        j['det_score'] = i[4]
                        overlapped_bboxes.append((index, j))
                        overlap = True
                        break
                
                if not overlap:
                    non_overlapped_bboxes.append(i)
                    new_kpss.append(kpss[index])
            
            bboxes = np.array(non_overlapped_bboxes)
            kpss = np.array(new_kpss)
            
        #print('overlapped boxes : ', len(overlapped_bboxes))
        #print('bboxes : ', bboxes.shape[0])
        #print('kpss : ', kpss.shape[0])
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            # print('face : ', face, '\n')
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                # print('model : ', model , '\n')
                model.get(img, face)
            ret.append(face)
            # print('ret : ', ret, '\n')

        # print('overlapped bbox : ', overlapped_bboxes)
        # print('overlapped bbox length : ' ,len(overlapped_bboxes))
        
        for i in overlapped_bboxes:
            ret.insert(i[0], i[1])
        
        # print('ret : ', ret)
        # print('ret length : ', len(ret))

        return ret

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(np.int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

            #for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(np.int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg

