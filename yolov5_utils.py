# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:29:12 2023

@author: Asus
"""

import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import time
import torchvision

def load_model(device):
    model = torch.jit.load("./640_yolos_wrapper.pth", map_location = device)
    print("Model Loaded")
    return model

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def draw_bbox(img,results):
    label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','Person']
    color=(0,255,255) 
    text = None
    for i in range(len(results)):
        result = results[i]
        text = label[int(result[5])]
        if text == 'Person': continue
        cv2.rectangle(img,(int(result[0]),int(result[1])),(int(result[2]),int(result[3])),color,2)
        score = str(round(result[4],2))
        # cv2.putText(img,(label[int(result[5])]+'-'+score),(int(result[0]),int(result[1])-2),cv2.FONT_HERSHEY_COMPLEX,0.25,color[int(result[5])],1)
        cv2.putText(img,(label[int(result[5])]+' conf-'+score),(int(result[0]),int(result[1])-3),cv2.FONT_HERSHEY_COMPLEX,0.5,color,1, cv2.LINE_AA)
    
    if text == 'Person': return img, ''
    return img, text

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter


def non_max_suppression(prediction, conf_thres=0.2, iou_thres=0.5, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def resize_img(img, size):
    h, w = img.shape[:2]
    if h < w:
        ratio = size/w
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
        diff = size-(h*ratio)
        t = int(diff/2)
        img = cv2.copyMakeBorder(img, t, t, 0, 0, cv2.BORDER_CONSTANT)

    else:
        ratio = size/h
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
        diff = size-(w*ratio)
        t = int(diff/2)
        # print(t)
        img = cv2.copyMakeBorder(img, 0, 0, t, t, cv2.BORDER_CONSTANT)

    img = cv2.resize(img,(size,size))

    flag = h < w
    return img, ratio, flag, t

def preprocess(img, size=640):
    '''
    Resize the image to the given size and convert it to tensor
    '''

    resized, ratio, flag, diff = resize_img(img, size)
    image = cv2.resize(resized, (size, size))
    return image, ratio, flag, diff

def give_yolo_result(img, model, device, size=640, confidence = 0.6): #input image is bgr image
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized, ratio, flag, diff = preprocess(img_rgb,size)
    resized = resized.transpose([2,0,1])/255.
    # resized = resized/255.
    resized = resized.astype(np.float32)[np.newaxis, ...]
    resized_tensor = torch.from_numpy(resized).to(device)
    # print(resized.shape)

    with torch.no_grad():
        z = model(resized_tensor)
    # print(z)
    # print(z.shape)
    a = z.view(25200,32)
    # b = a[a[:, 4].sort()[1]]
    # print(b)
    pred = non_max_suppression(
        z, confidence, 0.5, classes=None, agnostic=False)
    # print(pred)
    pred = pred[0].tolist()
    # print("preds==",pred)

    new_pred = []

    if flag:
        y_addition = int(diff/ratio)
        x_addition = 0
    else:
        y_addition = 0
        x_addition = int(diff/ratio)
    for p in pred:
        conf = round(p[4], 2)

        if conf > 0.6:
            x1 = int(p[0]/ratio - x_addition)
            y1 = int(p[1]/ratio - y_addition)
            x2 = int(p[2]/ratio - x_addition)
            y2 = int(p[3]/ratio - y_addition)

            cat = int(p[5])

            temp = [x1, y1, x2, y2, conf, cat]

            new_pred.append(temp)
    # yolo_total = round(time.time()-yolo_start,4)

    # if len(new_pred) == 0:
    
    #     abort(400, description="no-coin")

    predicted_img, text = draw_bbox(img,new_pred)
    return predicted_img, text