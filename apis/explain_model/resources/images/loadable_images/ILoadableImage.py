import abc
import pathlib

from pydantic import BaseModel
from apis.explain_model.resources.images.images import ModelImage
from apis.explain_model.resources.images.images import ModelImage
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.applications import imagenet_utils


class ILoadableImage(BaseModel, abc.ABC):
    path: pathlib.Path

    @abc.abstractmethod
    def load(self, input_dimensions) -> ModelImage:
        pass
