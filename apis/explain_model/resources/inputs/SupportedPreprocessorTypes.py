from enum import Enum


class SupportedPreprocessorTypes(str, Enum):
    ImageNet = "ImageNet"
    ResNet = "ResNet"