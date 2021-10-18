import torchvision
import torch
import argparse
import cv2
import detect_utils
from PIL import Image
import os
import torch.nn as nn
from train_torch import get_model_instance_segmentation

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/000016.jpg', help='path to input image/video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800,
                    help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())

# 保持和训练的时候模型一致
model = get_model_instance_segmentation(2)

# 加载训练好的参数文件
PATH = os.path.join(os.getcwd(), 'data/checkPoint/model.pt')
try:
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(PATH,map_location='cpu'),False)
    print("找到参数文件，加载参数预测！")
except FileNotFoundError:
    print("没有找到参数文件，采用原生模型预测！")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = Image.open(args['input'])
model.eval().to(device)
boxes, classes, labels = detect_utils.predict(image, model, device, 0.8)
image = detect_utils.draw_boxes(boxes, classes, labels, image)
cv2.imshow('Image', image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
cv2.imwrite(f"outputs/{save_name}.jpg", image)
cv2.waitKey(0)
