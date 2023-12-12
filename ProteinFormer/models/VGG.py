import torch
import torch.nn as nn


cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, num_class=7):
        super().__init__()
        self.features = features
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

#         self.classifier = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(1024, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(128, num_class)
#         )
        self.feature_layer = nn.Sequential(nn.Linear(512, 1024),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout(),
                                           nn.Linear(1024, 128),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout())
        self.last_layer = nn.Linear(128, num_class)
        self.init_weights()
        
    def init_weights(self, zero_init_last_bn=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def forward(self, x):
        output = self.features(x)
        output = self.avg(output)
        output = output.view(output.size()[0], -1)
#         print('output', output.shape)
#         output = self.classifier(output)
        out_feature = self.feature_layer(output)
        output = self.last_layer(out_feature)

        return output, out_feature

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def vgg11_bn(n_classes):
    return VGG(make_layers(cfg['A'], batch_norm=True), n_classes)

def vgg13_bn(n_classes):
    return VGG(make_layers(cfg['B'], batch_norm=True), n_classes)

def vgg16_bn(n_classes):
    return VGG(make_layers(cfg['D'], batch_norm=True), n_classes)

def vgg19_bn(n_classes):
    return VGG(make_layers(cfg['E'], batch_norm=True), n_classes)
