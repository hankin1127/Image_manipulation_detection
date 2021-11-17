import numpy as np
from torchvision.models import resnet50
import torch
from twoStreamNet.SRM import SRMLayer
from twoStreamNet.Rpn import RegionProposalNetwork
from torchvision.ops.roi_pool import RoIPool
from twoStreamNet.CompactBilinearPooling import CompactBilinearPooling
from torchvision.models.feature_extraction import create_feature_extractor
from twoStreamNet.Bbox_loss import compute_rpn_bbox_loss
from twoStreamNet.Utils import dictToOneTensor

m = resnet50()
return_nodes = {
    'layer1': 'layer1',
    'layer2': 'layer2',
    'layer3': 'layer3',
    'layer4': 'layer4'
}
create_feature_extractor(m, return_nodes=return_nodes)

class TwoStreamModel(torch.nn.Module):
    def __init__(self):
        super(TwoStreamModel, self).__init__()
        # SRM Filter
        self.srm = SRMLayer(channel=3)
        # resnet50 backbone
        m = resnet50()
        self.resnet50BackBone = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})
        # RPN
        self.rpn = RegionProposalNetwork(input_size=2048, layer_size=2, conv_size=3, num_anchor=3)
        # ROI POOLING
        # todo 此处初始化参数存疑
        self.roi = RoIPool(output_size=7, spatial_scale=5)
        # BilinearPooling
        self.bilinearPooling = CompactBilinearPooling(input_dim1=2048, input_dim2=2048, output_dim=1,cuda=False)
        # LOSS temper
        self.lossTemper = torch.nn.CrossEntropyLoss()

    def forward(self, rgbImage,targets):
        print(rgbImage.size())
        print(targets)
        # 1 得到图像噪声残差
        noiseImage = self.srm.forward(rgbImage)
        print(noiseImage.size())
        # 2 原图和噪声图分别提取特征
        rgbFeature = self.resnet50BackBone.forward(rgbImage)
        noiseFeature = self.resnet50BackBone.forward(noiseImage)
        print(rgbFeature['3'].size())
        print(noiseFeature['3'].size())
        # 3 原图提取建议框 此处取最高维度的特征图[2, 2048, 8, 8]
        classification_op, regression_op = self.rpn.forward(rgbFeature['3'])
        # 3.1 将两种特征融合
        classification_op = classification_op.unsqueeze(dim=1)
        rpnResult = torch.cat([classification_op, regression_op], dim=1)
        print(rpnResult.size())
        # 3.2 计算bboxLoss
        # bboxLoss = compute_rpn_bbox_loss()

        # 4 获取ROI POOLING 后的特征
        noiseRoiFeature = self.roi.forward(noiseFeature['3'], rpnResult)
        print(noiseRoiFeature.size())
        rgbRoiFeature = self.roi.forward(rgbFeature['3'], rpnResult)
        print(rgbRoiFeature.size())

        # 5 双线性池化
        bilinearPoolingOutput = self.bilinearPooling.forward(rgbRoiFeature,noiseRoiFeature)
        bilinearPoolingOutput = torch.reshape(bilinearPoolingOutput, (1, 192))
        print(bilinearPoolingOutput.size())
        # 5.1 计算LOSS tamper(为什么想到用这个交叉熵损失函数)
        loss_temper = self.lossTemper.forward(bilinearPoolingOutput,targets[0]['labels'])
        print(loss_temper)
        return loss_temper