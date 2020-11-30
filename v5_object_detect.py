import sys
import os
import numpy as np
import acl
import cv2 
from PIL import Image

from constants import *
from acl_resource import AclResource
from utils import *
from acl_model import Model
from acl_image import AclImage
import time
import pdb
labels = ["airplane", "ship", "oiltank", "playground", "port", "bridge", "car"]

INPUT_DIR = './data/'
OUTPUT_DIR = './out_od/'   
OUTPUT_TXT_DIR = './mAP/predicted'
MODEL_PATH = "yolov5_0828.om"
MODEL_WIDTH = 640
MODEL_HEIGHT = 640

stride_list = [8, 16, 32]
conf_threshold = 0.01
iou_threshold = 0.5
class_num = len(labels)
num_channel = 3 * (class_num + 5)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
na, no = 3, 12  #na anchor number， no=nc+5
stride = [8.0,16.0,32.0]
anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
anchors = np.array(anchors).astype(np.float).reshape(3, -1, 2)
anchor_grid = anchors.reshape(3, 1, -1, 1, 1, 2)
grids = []
for idx in range(3):
    grids.append(np.load("/home/HwHiAiUser/OD/grid/grid0828_{}.npy".format(idx)))

def letterbox(img, new_shape=(320, 640), color=(114, 114, 114)):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    dw, dh = 0.0, 0.0
    new_unpad = (new_shape[1], new_shape[0])
    ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])  # width, height ratios

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    return img, ratio

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

def convert_labels(label_list):
    if isinstance(label_list, np.ndarray):
        label_list = label_list.tolist()
        label_names = [labels[int(index)] for index in label_list]
    return label_names

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:,0] = x[:,0] - x[:,2] / 2  # top left x
    y[:,1] = x[:,1] - x[:,3] / 2  # top left y
    y[:,2] = x[:,0] + x[:,2] / 2  # bottom right x
    y[:,3] = x[:,1] + x[:,3] / 2  # bottom right y
    return y
    
def clip_coords(boxes, img_shape):
    boxes[:,0] = boxes[:,0].clip(0, img_shape[1])  # x1
    boxes[:,1] = boxes[:,1].clip(0, img_shape[0])  # y1
    boxes[:,2] = boxes[:,2].clip(0, img_shape[1])  # x2
    boxes[:,3] = boxes[:,3].clip(0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio):
    #pdb.set_trace()
    coords[:,[0, 2]] /= ratio[0]  # divide ratio
    coords[:,[1, 3]] /= ratio[1]
    clip_coords(coords, img0_shape)
    return coords
    
def yolov5_post(xs):
    z = []
    nynxs= [[40,80], [20,40],[10,20]]
    for idx, (ny, nx) in enumerate(nynxs):
        x = xs[idx]
        x = x.reshape(1, na, no, ny, nx)
        x = x.transpose(0, 1, 3, 4, 2)
        y = sigmoid(x)
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grids[idx]) * stride[idx]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[idx]
        z.append(y.reshape(1, -1, 12))
    res = np.concatenate(z, 1)
    return res#[1,25200,12]

def non_max_suppression(prediction, conf_thres=0.01,iou_thres = 0.5):
    nc = prediction[0].shape[1] - 5  # number of classes 7
    xc = prediction[..., 4] > conf_thres  # candidates [1,12600] bool 
    
    t = time.time()
    x = prediction[0] #[25200,12]
    x = x[xc[0]]  # confidence [36,12] #36是挑选出来的框
    # Compute conf
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[:, :4])#[36,4]

    conf_ = x[:,5:].max(1) #7个类置信度的最大值 1表示是这一行  ,keepdims=True保持维度是[xxx,1]  (30,)
    conf= conf_[:,np.newaxis] #(30,1)
    j = x[:,5:].argmax(-1) #cls
    j = j[:,np.newaxis]#(30,1)

    pred = np.concatenate((box,conf,j),1)[conf_> conf_thres]#(12,6) 12是box的个数
    all_boxes = [[] for ix in range(class_num)] 
    for ix in range(pred.shape[0]): #所有选出来的box
        bbox = [int(pred[ix,iy])for iy in range(4)]
        bbox.append(int(pred[ix,5]))#cls
        bbox.append(pred[ix,4])
        all_boxes[bbox[4]-1].append(bbox) #这个类对应的bbox
    
    res = apply_nms(all_boxes, iou_thres)
    result_return = dict()
    if not res:
        result_return['detection_classes'] = []
        result_return['detection_boxes'] = []
        result_return['detection_scores'] = []
        return result_return
    else:
        new_res = np.array(res)
        picked_boxes = new_res[:, 0:4]
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
    for pic in src_dir:
        #读取图片
        pic_path = os.path.join(INPUT_DIR, pic)
        pic_name = pic.split('.')[0]
        print(pic_name)
        bgr_img = cv2.imread(pic_path)
        
        t1 = time.time()
        img, ratio = letterbox(bgr_img, new_shape=(320,640)) # resize to (320,640,3)
        img = img[:,:,::-1]#bgr to rgb
        img = img.transpose(2,0,1)#(3,320,640)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img = img / 255.0

        data = np.concatenate((img[:,::2,::2],img[:,1::2,::2],img[:,::2,1::2],img[:,1::2,1::2]), axis=0)#[12,160,320]
        t2 = time.time()
        result_list = model.execute([data,]) 
        t3 = time.time()   
        post = yolov5_post(result_list)#[1,25200,12]
        result_return = non_max_suppression(post,conf_thres = conf_threshold,iou_thres = iou_threshold)
        if len(result_return['detection_classes']):
            det = np.array( result_return['detection_boxes'])[:,:4]
            bbox = scale_coords((320, 640), det,bgr_img.shape, ratio)
        t4 = time.time()
        print("result = ", result_return)
        print("preprocess cost：", t2-t1)
        print("forward cost：", t3-t2)
        print("postprocess cost：", t4-t3)
        print("total cost：", t4-t1)
        print("FPS：", 1/(t4-t1))

        for i in range(len(result_return['detection_classes'])):
            box = bbox[i]
            class_name = result_return['detection_classes'][i]
            confidence = result_return['detection_scores'][i]
            cv2.rectangle(bgr_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors[i%6])
            p3 = (max(int(box[0]), 15), max(int(box[1]), 15))
            out_label = class_name            
            cv2.putText(bgr_img, out_label, p3, cv2.FONT_ITALIC, 0.6, colors[i%6], 1)
        output_file = os.path.join(OUTPUT_DIR, "out_" + pic)
        print("output:%s" % output_file)
        cv2.imwrite(output_file, bgr_img)

        pic_name = pic.split('.')[0]
        predict_result_path = os.path.join(OUTPUT_TXT_DIR , str(pic_name)+'.txt')
        with open(predict_result_path, 'w') as f:
            for i in range(len(result_return['detection_classes'])):
                box = bbox[i]
                class_name = result_return['detection_classes'][i]
                confidence = result_return['detection_scores'][i]
                box = list(map(int, box))
                box = list(map(str, box))
                confidence = '%.4f' % confidence
                bbox_mess = ' '.join([class_name, confidence, box[0], box[1], box[2], box[3]]) + '\n'
                f.write(bbox_mess)
    print("Execute end")

if __name__ == '__main__':
    main()
 
