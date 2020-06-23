import torch
import numpy as np
import os
import json
from PIL import Image
import api
import torchvision.transforms as transforms
from detector import Detector
from utils_box.dataset import show_bbox
import time, datetime

# Read train.json and set current GPU (for nms_cuda) and prepare the network
with open('train.json', 'r') as load_f:
    cfg = json.load(load_f)
net = Detector(pretrained=False)
net.load_state_dict(torch.load('net.pkl', map_location='cpu'))
net.eval()
dir = "images"

# TODO: Set nms_th
net.nms_th = 0.5
# ==================================


# Read LABEL_NAMES
with open(cfg['name_file']) as f:
    lines = f.readlines()
LABEL_NAMES = []
for line in lines:
    LABEL_NAMES.append(line.strip())

# Prepare API structure

from utils_box.dataset import center_fix
from detector import get_loss, get_pred


class Inferencer(object):
    def __init__(self, net):
        '''
        external initialization structure:
            net(Model)
        '''
        self.net = net
        self.normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def pred(self, img_pil):
        '''
        all models should be in cuda()
        return cls_i_preds, cls_p_preds, reg_preds
        '''
        _boxes = torch.zeros(0, 4)
        img_pil, boxes, loc, scale = center_fix(img_pil, _boxes, self.net.view_size)
        img = transforms.ToTensor()(img_pil)
        img = self.normalizer(img).view(1, img.shape[0], img.shape[1], img.shape[2])
        img = img
        loc = loc.view(1, -1)
        with torch.no_grad():
            temp = self.net(img, loc)
            cls_i_preds, cls_p_preds, reg_preds = get_pred(temp,
                                                           self.net.nms_th, self.net.nms_iou)
            reg_preds[0][:, 0::2] -= loc[0, 0]
            reg_preds[0][:, 1::2] -= loc[0, 1]
            reg_preds[0] /= scale
        return cls_i_preds[0], cls_p_preds[0], reg_preds[0]


inferencer = Inferencer(net)

# Run
for filename in os.listdir(dir):
    if filename.endswith('.jpg'):
        img = Image.open(os.path.join(dir, filename))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        prev_t = time.time()
        cls_i_preds, cls_p_preds, reg_preds = inferencer.pred(img)
        cur_t = time.time()
        inference_time = datetime.timedelta(seconds=cur_t - prev_t)
        print("\t+ Inference Time: %s" % (inference_time))

        name = dir + '/pred_' + filename.split('.')[0]

        show_bbox(img, reg_preds.cpu(), cls_i_preds.cpu(), cls_p_preds.cpu(), LABEL_NAMES, name)
        print(filename, "done")
