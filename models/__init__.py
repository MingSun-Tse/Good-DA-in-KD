from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50, ResNet18
from .resnet_imagenet import resnet18
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2, wrn_28_10
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg13_bn_3neurons, vgg11_bn, vgg8_bn, vgg8_bn_3neurons
from .mobilenetv2 import mobile_half, mobile_0_25
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_0_3

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'ResNet18': ResNet18,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_28_10': wrn_28_10,
    'vgg8': vgg8_bn,
    'vgg8_3neurons': vgg8_bn_3neurons,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg13_3neurons': vgg13_bn_3neurons,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'MobileNetV2_0_25': mobile_0_25,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'ShuffleV2_0_3': ShuffleV2_0_3,
    'resnet18': resnet18,
}
