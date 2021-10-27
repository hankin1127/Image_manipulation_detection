import torch
import torch.nn as nn


class RegionProposalNetwork(nn.Module):
    def __init__(self, input_size, layer_size, conv_size, num_anchor):
        super().__init__()

        self.input_size = input_size
        self.layer_size = layer_size
        self.num_anchor = num_anchor
        self.conv_size = conv_size

        self.intermediate = nn.Conv2d(
            self.input_size, self.layer_size, self.conv_size, stride=1, padding=1
        )
        self.classification_head = nn.Conv2d(self.layer_size, self.num_anchor, 1)
        self.reggresion_head = nn.Conv2d(self.layer_size, 4 * self.num_anchor, 1)

        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, feature_map):

        t = torch.nn.functional.relu(self.intermediate(feature_map))
        classification_op = self.classification_head(t)
        regression_op = self.reggresion_head(t)

        classification_op = classification_op.permute(0,2,3,1).flatten()
        regression_op = regression_op.permute(0,2,3,1).reshape(-1,4)

        return classification_op, regression_op