import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DenseNetBackbone(nn.Module):
    def __init__(self):
        super(DenseNetBackbone, self).__init__()
        densenet = models.densenet121(models.DenseNet121_Weights.DEFAULT)

        self.features = densenet.features

    def forward(self, x):
        features = []
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in ["conv0", "denseblock1", "denseblock2", "denseblock3"]:
                features.append(x)
        return features

class BiFPNLayer(nn.Module):
    def __init__(self, channels, epsilon=1e-4):
        super(BiFPNLayer, self).__init__()
        self.epsilon = epsilon

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

    def forward(self, inputs):

        P3, P4, P5, P6 = inputs

        w = F.relu(self.w1)
        weight = w / (torch.sum(w) + self.epsilon)
        P5_td = self.conv(
            weight[0] * P5
            + weight[1] * F.interpolate(P6, size=P5.shape[-2:], mode="bicubic")
        )

        w = F.relu(self.w1)
        weight = w / (torch.sum(w) + self.epsilon)
        P4_td = self.conv(
            weight[0] * P4
            + weight[1] * F.interpolate(P5_td, size=P4.shape[-2:], mode="bicubic")
        )

        w = F.relu(self.w1)
        weight = w / (torch.sum(w) + self.epsilon)
        P3_td = self.conv(
            weight[0] * P3
            + weight[1] * F.interpolate(P4_td, size=P3.shape[-2:], mode="bicubic")
        )

        # Bottom-up: downsample and fuse
        w = F.relu(self.w2)
        weight = w / (torch.sum(w) + self.epsilon)
        P4_out = self.conv(
            weight[0] * P4
            + weight[1] * P4_td
            + weight[2] * F.max_pool2d(P3_td, kernel_size=2)
        )

        w = F.relu(self.w2)
        weight = w / (torch.sum(w) + self.epsilon)
        P5_out = self.conv(
            weight[0] * P5
            + weight[1] * P5_td
            + weight[2] * F.max_pool2d(P4_out, kernel_size=2)
        )

        w = F.relu(self.w2)
        weight = w / (torch.sum(w) + self.epsilon)
        P6_out = self.conv(
            weight[0] * P6
            + weight[1] * F.max_pool2d(P5_out, kernel_size=2)
            + weight[2] * P6
        )

        return [P3_td, P4_out, P5_out, P6_out]


class BiFPN(nn.Module):
    def __init__(self, channels, num_layers=2):
        super(BiFPN, self).__init__()
        self.layers = nn.ModuleList([BiFPNLayer(channels) for _ in range(num_layers)])

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class CenterNetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CenterNetHead, self).__init__()
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1),
        )

    def forward(self, x):
        heatmap = torch.sigmoid(self.heatmap_head(x))
        size = torch.sigmoid(self.size_head(x))
        offset = torch.tanh(self.offset_head(x))
        return heatmap, size, offset


class ObjectDetectionModel(nn.Module):
    def __init__(self, tp_class = 8, tc_class = 2, bifpn_channels=256):
        super(ObjectDetectionModel, self).__init__()
        self.backbone = DenseNetBackbone()

        self.adapter_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, bifpn_channels, kernel_size=1)
                for in_channels in [ 64, 256, 512, 1024]
            ]
        )

        self.bifpn = BiFPN(bifpn_channels, num_layers=2)

        self.tc_head = CenterNetHead(bifpn_channels, tc_class) # traffic control objects
        self.tp_head = CenterNetHead(bifpn_channels, tp_class) # traffic participants objects

    def forward(self, x):

        feats = self.backbone(x)
        feats_adapted = []
        for feat, conv in zip(feats, self.adapter_convs):
            feats_adapted.append(conv(feat))
        fused_feats = self.bifpn(feats_adapted)
        traffic_control_objects, traffic_participant_objects = fused_feats[0], fused_feats[1]
        tc_heatmap, tc_size, tc_offset = self.tc_head(traffic_control_objects)
        tp_heatmap, tp_size, tp_offset = self.tp_head(traffic_participant_objects)
        return {"tc":{"heatmap": tc_heatmap, "size": tc_size, "offset": tc_offset},
                "tp":{"heatmap": tp_heatmap, "size": tp_size, "offset": tp_offset}}


if __name__ == "__main__":
    dummy_input = torch.randn(
        1, 3, 720, 1280
    )  # dummy input tensor [batch, channels, height, width]
    tp_classes = 8
    tc_classes = 2

    model = ObjectDetectionModel(
        tp_class = tp_classes, tc_class = tc_classes, bifpn_channels=256
    )
    outputs = model(dummy_input)

    print("Heatmap shape:", outputs["tc"]["heatmap"].shape)
    print("Size shape:", outputs["tc"]["size"].shape)
    print("Offset shape:", outputs["tc"]["offset"].shape)
    print("-"*50)
    print("Heatmap shape:", outputs["tp"]["heatmap"].shape)
    print("Size shape:", outputs["tp"]["size"].shape)
    print("Offset shape:", outputs["tp"]["offset"].shape)
