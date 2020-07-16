import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['cnn_resnet50_center']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class cnn_resnet_center(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(cnn_resnet_center, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.maxpool_2 = nn.MaxPool2d(kernel_size=7, stride=1, return_indices=True)  # First modification
        
        self.fc = nn.Linear(512 * block.expansion, 1000)
        self.fc_softmax = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

   

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x_conv1 = self.relu(x)
        maxpool_1_size = x.shape
        x, indices_1 = self.maxpool_1(x_conv1)

        x_l1 = self.layer1(x)
        
        x_l2 = self.layer2(x_l1)
        x_l3 = self.layer3(x_l2)
        x_l4 = self.layer4(x_l3)

        x, indices_2 = self.maxpool_2(x_l4)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x_softmax = self.fc_softmax(x)

        return x, x_softmax



def cnn_resnet50_center(pretrained=False,  num_classes=1000, embedding_size=128, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = cnn_resnet_center(Bottleneck, [3, 4, 6, 3], num_classes=1000,**kwargs)


    weights_imagenet = model_zoo.load_url(model_urls['resnet50'])
    weights_imagenet["fc_softmax.weight"] = weights_imagenet["fc.weight"]
    weights_imagenet["fc_softmax.bias"] = weights_imagenet["fc.bias"]

    if pretrained:
        model.load_state_dict(weights_imagenet)

    model.fc = nn.Linear(2048, embedding_size)
    model.fc_softmax = nn.Linear(embedding_size, num_classes)

    return model