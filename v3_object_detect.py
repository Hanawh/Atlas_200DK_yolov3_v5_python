import sys
import os
import numpy as np
import acl
import cv2 as cv
from PIL import Image

from constants import *
from acl_resource import AclResource
from utils import *
from acl_model import Model
from acl_image import AclImage
import time
labels = ["airplane", "ship", "oiltank", "playground", "port", "bridge", "car"]

INPUT_DIR = './test_data/'
OUTPUT_DIR = './out_od/'   
OUTPUT_TXT_DIR = './mAP/predicted'
MODEL_PATH = "yolov3_1128.om"
MODEL_WIDTH = 416
MODEL_HEIGHT = 416

stride_list = [8, 16, 32]

conf_threshold = 0.3
iou_threshold = 0.45
class_num = len(labels)
num_channel = 3 * (class_num + 5)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

def preprocess(img_path): 
    image = Image.open(img_path)#(985, 769)  w h
    img_h = image.size[1]#769
    img_w = image.size[0]#985
    net_h = MODEL_HEIGHT#640
    net_w = MODEL_WIDTH

    scale = min(float(net_w) / float(img_w), float(net_h) / float(img_h))
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    shift_x = (net_w - new_w) // 2
    shift_y = (net_h - new_h) // 2

    image_ = image.resize( (new_w, new_h))
    new_image = np.zeros((net_h, net_w, 3), np.uint8)
    new_image[shift_y: new_h + shift_y, shift_x: new_w + shift_x, :] = np.array(image_)
    new_image = new_image.astype(np.float32)
    new_image = new_image / 255
    return new_image, img_w, img_h

def preprocess_cv2(bgr_img):
    shape = bgr_img.shape[:2]  # [height, width]

    net_h = MODEL_HEIGHT
    net_w = MODEL_WIDTH
    scale= min(float(net_h) / float(shape[0]), float(net_w) / float(shape[1]))
    new_h = int(shape[0] * scale)
    new_w = int(shape[1] * scale)
    
    dw = (net_w - new_w) / 2
    dh = (net_h - new_h) / 2

    img = cv.resize(bgr_img,  (new_w, new_h), interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))  
    '''
    print("top :" ,top)
    print("bottom :" ,bottom)
    print("left:" ,left)
    print("right :" ,right)
    '''
    #img = cv.resize(bgr_img,  (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv.INTER_LINEAR) 
    #img = img.astype(np.float32)
    #img = img / 255.0
    return img, shape[1],shape[0]


def overlap(x1, x2, x3, x4):
    left = max(x1, x3)
    right = min(x2, x4)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(all_boxes, thres):
    res = []

    for cls in range(class_num):
        cls_bboxes = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]

        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue

            truth = sorted_boxes[i]
            for j in range(i + 1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1

        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res

def decode(conv_output,  img_w, img_h):
    h, w, _ = conv_output.shape
    pred = conv_output.reshape((h * w, 3, 5 + class_num))
    print(pred[0][0])
    resize_ratio = min(MODEL_WIDTH / img_w,  MODEL_HEIGHT / img_h)
    dw = (MODEL_WIDTH - resize_ratio * img_w) / 2
    dh = (MODEL_HEIGHT - resize_ratio * img_h) / 2

    bbox = np.zeros((h * w, 3, 4))
    bbox[..., 0] = np.maximum((pred[..., 0] - pred[..., 2] / 2.0 - dw) / resize_ratio, 0)  # x_min
    bbox[..., 1] = np.maximum((pred[..., 1] - pred[..., 3] / 2.0 - dh) / resize_ratio, 0)  # y_min
    bbox[..., 2] = np.minimum((pred[..., 0] + pred[..., 2] / 2.0 - dw)/ resize_ratio, img_w)  # x_max
    bbox[..., 3] = np.minimum((pred[..., 1] + pred[..., 3] / 2.0 - dh) / resize_ratio, img_h)  # y_max
    
    pred[..., :4] = bbox
    pred = pred.reshape((-1, 5 + class_num))
    pred[:, 4] = pred[:, 4] * pred[:, 5:].max(1)
    pred = pred[pred[:, 4] >= conf_threshold]
    pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)

    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):
        box = [int(pred[ix, iy]) for iy in range(4)]
        box.append(int(pred[ix, 5]))
        box.append(pred[ix, 4])
        all_boxes[box[4] - 1].append(box)
    return all_boxes

def overlap(x1, x2, x3, x4):
    left = max(x1, x3)
    right = min(x2, x4)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(all_boxes, thres):
    res = []

    for cls in range(class_num):
        cls_bboxes = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]

        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue

            truth = sorted_boxes[i]
            for j in range(i + 1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1

        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res

def decode(conv_output,  img_w, img_h):
    h, w, _ = conv_output.shape
    pred = conv_output.reshape((h * w, 3, 5 + class_num))
    resize_ratio = min(MODEL_WIDTH / img_w,  MODEL_HEIGHT / img_h)
    dw = (MODEL_WIDTH - resize_ratio * img_w) / 2
    dh = (MODEL_HEIGHT - resize_ratio * img_h) / 2

    bbox = np.zeros((h * w, 3, 4))
    bbox[..., 0] = np.maximum((pred[..., 0] - pred[..., 2] / 2.0 - dw) / resize_ratio, 0)  # x_min
    bbox[..., 1] = np.maximum((pred[..., 1] - pred[..., 3] / 2.0 - dh) / resize_ratio, 0)  # y_min
    bbox[..., 2] = np.minimum((pred[..., 0] + pred[..., 2] / 2.0 - dw)/ resize_ratio, img_w)  # x_max
    bbox[..., 3] = np.minimum((pred[..., 1] + pred[..., 3] / 2.0 - dh) / resize_ratio, img_h)  # y_max
    
    pred[..., :4] = bbox
    pred = pred.reshape((-1, 5 + class_num))
    pred[:, 4] = pred[:, 4] * pred[:, 5:].max(1)
    pred = pred[pred[:, 4] >= conf_threshold]
    pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)
    print(pred[:,:5])
    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):
        box = [int(pred[ix, iy]) for iy in range(4)]
        box.append(int(pred[ix, 5]))
        box.append(pred[ix, 4])
        all_boxes[box[4] - 1].append(box)
    return all_boxes

def convert_labels(label_list):
    if isinstance(label_list, np.ndarray):
        label_list = label_list.tolist()
        label_names = [labels[int(index)] for index in label_list]
    return label_names

    if isinstance(label_list, np.ndarray):
        label_list = label_list.tolist()
        label_names = [labels[int(index)] for index in label_list]
    return label_names

def post_process(infer_output, img_w, img_h):
    print("post process")
    result_return = dict()
    all_boxes = [[] for ix in range(class_num)]
    for ix in range(3):
        pred = infer_output[ix].reshape((MODEL_HEIGHT // stride_list[ix], MODEL_WIDTH // stride_list[ix], num_channel))
        boxes = decode(pred,  img_w, img_h)
        all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(class_num)]
    res = apply_nms(all_boxes, iou_threshold)
    if not res:
        result_return['detection_classes'] = []
        result_return['detection_boxes'] = []
        result_return['detection_scores'] = []
        return result_return
    else:
        new_res = np.array(res)
        picked_boxes = new_res[:, 0:4]
        picked_boxes = picked_boxes[:, [1, 0, 3, 2]]
        picked_classes = convert_labels(new_res[:, 4])
        picked_score = new_res[:, 5]
        result_return['detection_classes'] = picked_classes
        result_return['detection_boxes'] = picked_boxes.tolist()
        result_return['detection_scores'] = picked_score.tolist()
        return result_return

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_TXT_DIR):
        os.mkdir(OUTPUT_TXT_DIR)
    #acl资源初始化
    acl_resource = AclResource()
    acl_resource.init()
    #加载模型
    model = Model(acl_resource, MODEL_PATH)
    src_dir = os.listdir(INPUT_DIR)
    print("src_dir = ", src_dir)
    #从data目录逐张读取图片进行推理

    t_pre = 0
    t_for = 0
    t_post = 0

    for pic in src_dir:
        #读取图片
        pic_path = os.path.join(INPUT_DIR, pic)
        bgr_img = cv.imread(pic_path)
        #预处理
        t1 = time.time()
        #data, w, h= preprocess(pic_path)
        data, w, h = preprocess_cv2(bgr_img)
        t2 = time.time()
        t_pre += (t2-t1)
        #送进模型推理
        result_list = model.execute([data,]) 
        t3 = time.time()   
        t_for += (t3-t2)
        #处理推理结果
        result_return = post_process(result_list, w, h)
        t4 = time.time()
        t_post += (t4-t3)
        print("result = ", result_return)
        print("preprocess cost：", t2-t1)
        print("forward cost：", t3-t2)
        print("proprocess cost：", t4-t3)
        for i in range(len(result_return['detection_classes'])):
            box = result_return['detection_boxes'][i]
            class_name = result_return['detection_classes'][i]
            confidence = result_return['detection_scores'][i]
            cv.rectangle(bgr_img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), colors[i%6], 2)
            p3 = (max(int(box[1]), 15), max(int(box[0]), 15))
            out_label = class_name            
            cv.putText(bgr_img, out_label, p3, cv.FONT_ITALIC, 0.6, colors[i%6], 1)
        output_file = os.path.join(OUTPUT_DIR, "out_" + pic)
        print("output:%s" % output_file)
        cv.imwrite(output_file, bgr_img)

        pic_name = pic.split('.')[0]
        predict_result_path = os.path.join(OUTPUT_TXT_DIR, str(pic_name)+'.txt')
        with open(predict_result_path, 'w') as f:
            for i in range(len(result_return['detection_classes'])):
                box = result_return['detection_boxes'][i]
                class_name = result_return['detection_classes'][i]
                confidence = result_return['detection_scores'][i]
                box = list(map(int, box))
                box = list(map(str, box))
                confidence = '%.4f' % confidence
                bbox_mess = ' '.join([class_name, confidence, box[1], box[0], box[3], box[2]]) + '\n'
                f.write(bbox_mess)
    num = len(src_dir)
    print("avg preprocess cost：", t_pre/num)
    print("avg forward cost：", t_for/num)
    print("avg proprocess cost：", t_post/num)
    total = t_pre/num + t_for/num +  t_post/num
    print("avg total cost：", total)
    print("avg FPS：", 1/(total))
    print("Execute end")

if __name__ == '__main__':
    main()
 
