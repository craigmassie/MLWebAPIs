from enum import Enum


class SupportedLibraryTypes(str, Enum):
    Keras = "Keras"
    Tensorflow = "Tensorflow"