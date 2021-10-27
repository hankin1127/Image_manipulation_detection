import numpy as np
from torchvision.models import resnet50
import torch
from twoStreamNet.SRM import SRMLayer
from twoStreamNet.Rpn import RegionProposalNetwork
from torchvision.ops.roi_pool import RoIPool
from twoStreamNet.CompactBilinearPooling import CompactBilinearPooling


class TwoStreamModel(torch.nn.Module):
    def __init__(self):
        super(TwoStreamModel, self).__init__()
        # SRM Filter
        self.srm = SRMLayer(channel=3)
        # resnet50 backbone
        self.resnet50BackBone = resnet50()
        # RPN
        # todo 此处初始化参数存疑
        self.rpn = RegionProposalNetwork(input_size=2048, layer_size=3*2048, conv_size=3, num_anchor=3)
        # ROI POOLING
        # todo 此处初始化参数存疑
        output_size_roi = 7
        self.roi = RoIPool(output_size=output_size_roi, spatial_scale=5)
        # BilinearPooling
        # todo 此处初始化参数存疑
        self.bilinearPooling = CompactBilinearPooling(input_dim1=output_size_roi, input_dim2=output_size_roi, output_dim=2,cuda=False)

    def forward(self, rgbImage,targets):
        # 得到图像噪声残差
        noiseImage = self.srm.forward(rgbImage)
        # 原图和噪声图分别提取特征
        rgbFeature = self.resnet50BackBone.forward(rgbImage)
        noiseFeature = self.resnet50BackBone.forward(noiseImage)
        # 原图提取建议框
        classification_op, regression_op = self.rpn.forward(rgbFeature)
        # 获取ROI POOLING 后的特征
        rgbRoiFeature = self.roi.forward(rgbFeature, regression_op) # todo 此处初始化参数存疑
        noiseRoiFeature = self.roi.forward(noiseFeature) # todo 此处初始化参数存疑
        # 双线性池化
        output = self.bilinearPooling.forward(rgbRoiFeature,noiseRoiFeature)
        return output