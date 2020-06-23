import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            FrozenBatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            FrozenBatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(out_channels)
            FrozenBatchNorm2d(out_channels),
        )

        if in_channels == out_channels:  # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else:  # when dim change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channels)
                FrozenBatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        residual = self.shortcut(residual)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += residual

        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, num_classes, num_block_lists=[3, 4, 6, 3]):
        super(ResNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            # nn.BatchNorm2d(64),
            FrozenBatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage_1 = self._make_layer(64, 64, 256, nums_block=num_block_lists[0], stride=1)
        self.stage_2 = self._make_layer(256, 128, 512, nums_block=num_block_lists[1], stride=2)
        self.stage_3 = self._make_layer(512, 256, 1024, nums_block=num_block_lists[2], stride=2)
        self.stage_4 = self._make_layer(1024, 512, 2048, nums_block=num_block_lists[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, in_channels, mid_channels, out_channels, nums_block, stride=1):
        layers = [Bottleneck(in_channels, mid_channels, out_channels, stride=stride)]
        for _ in range(1, nums_block):
            layers.append(Bottleneck(out_channels, mid_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, classify=False):
        x = self.basic_conv(x)
        x = self.stage_1(x)
        out3 = self.stage_2(x)
        out4 = self.stage_3(out3)
        out5 = self.stage_4(out4)

        if classify:
            x = self.gap(out5)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
        else:
            return (out3, out4, out5)

    def freeze_bn(self):
        pass
        # for layer in self.modules():
        # if isinstance(layer, nn.BatchNorm2d):
        # layer.eval()

    def freeze_stages(self, stage):
        if stage >= 0:
            self.basic_conv[1].eval()
            for m in [self.basic_conv[0], self.basic_conv[1]]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, stage + 1):
            layer = getattr(self, 'stage_{}'.format(i))
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    from collections import OrderedDict
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def resnet50(pretrained=True, model_path='best_model.pkl'):
    model = ResNet(1000, [3, 4, 6, 3])
    if pretrained:
        checkpoint = torch.load(model_path, map_location='cpu')
        state = checkpoint["state_dict"]
        state = convert_state_dict(state)
        model.load_state_dict(state, strict=True)
        print('load model done!')
    return model


if __name__ == '__main__':
    print('load model')

    print(resnet50())
