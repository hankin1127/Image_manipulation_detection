# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import torch
from PIL import Image
import math
import sys

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import xml.etree.ElementTree as ET
import time

# from engine import train_one_epoch, evaluate
import utils
import transforms as T
import numpy as np

class HandDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Annotations"))))

    # 取Dataset中的第idx个元素
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "Annotations", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        # masks = np.load(mask_path)

        # 解析XML文件
        tree = ET.parse(mask_path)
        root = tree.getroot()

        boxes = [[int(root[-1][1][i].text) for i in range(4)]]
        # h, w = masks.shape
        # masks = masks.reshape(1, h, w)

        # 假设每张训练的图中只有一个篡改物体
        num_objs = 1
        # boxes = []
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# def get_model_instance_segmentation(num_classes):
#     # load an instance segmentation model pre-trained pre-trained on COCO
#     model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#
#     # get number of input features for the classifier
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     # replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#
#     # now get the number of input features for the mask classifier
#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#     hidden_layer = 256
#     # and replace the mask predictor with a new one
#     model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
#
#     return model


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def get_transform(train):
    transforms = []
    # transforms.ToTensor把数据处理成[0,1]，每个像素除255
    transforms.append(T.ToTensor())
    if train:
        # 依据概率p对PIL图片进行水平翻转
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    header = 'Test:'

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes

    # accumulate predictions from all images


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and 篡改物体
    num_classes = 2
    # use our dataset and defined transformations
    root_dir = os.path.join(os.getcwd(), 'data/DIY_dataset/VOC2007')
    dataset = HandDataset(root_dir, get_transform(train=True))
    dataset_test = HandDataset(root_dir, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-10])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-10:])

    num_workers = 4  # Default to 4
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                              num_workers=num_workers, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False,
                                                   num_workers=num_workers, collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    # 训练参数保存路径
    checkPointPath = os.path.join(os.getcwd(), 'data/checkPoint/model.pt')
    # 损失函数列表
    lossList = []
    # 检查是否已经有训练参数，如有继续训练
    try:
        if(device.type == 'cpu'):
            checkpoint = torch.load(checkPointPath, map_location='cpu')
        else:
            checkpoint = torch.load(checkPointPath)
        # 模型和优化器参数
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # loss存储列表
        lossList = checkpoint['loss']
        # 剩余epoch数
        leftEpoch = checkpoint['leftEpoch']
        if isinstance(leftEpoch, int):
            num_epochs = leftEpoch
        model.eval()
        print("找到参数文件，开始接着训练！一共训练"+str(num_epochs)+"轮")
    except FileNotFoundError:
        print("没有找到参数文件，重新开始训练！一共训练"+str(num_epochs)+"轮")


    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        # Train function integrate here.    ------------------------
        model.train()
        # metric_logger = utils.MetricLogger(delimiter="  ")
        # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        # header = 'Epoch: [{}]'.format(epoch)
        print("Epoch轮次：" + str(epoch) + "，开始=======")
        # for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        i = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = utils.reduce_dict(loss_dict)
            # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            # losses_reduced = sum(loss for loss in losses.values())

            # loss_value = losses_reduced.item()
            loss_value = losses.item()
            print(loss_value)
            # 每次训练，记录模型的loss值
            lossList.append(loss_value)

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                # print(loss_dict_reduced)
                print(losses)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()


            if(i % 500 == 0):
                # 每个500次训练保存一下参数
                torch.save({
                'leftEpoch': num_epochs - epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': lossList,
                }, checkPointPath)
                print("Epoch轮次：" + str(epoch) +"，第"+str(i)+ "次训练，保存训练参数")

            i = i+1

    print("That's it!")

    # with torch.no_grad():


if __name__ == "__main__":
    main()
