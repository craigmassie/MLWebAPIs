import abc
from typing import Tuple

import typing

from apis.explain_model.resources.inputs.images import ModelImage


class Model(abc.ABC):
    type: str

    def __init__(self):
        # ...
        pass

    @property
    @abc.abstractmethod
    def model(self):
        pass

    @property
    @abc.abstractmethod
    def input_dimensions(self):
        pass

    @abc.abstractmethod
    def explain_image_instance(self, image: ModelImage) -> Tuple[ModelImage.ModelImage, typing.Any]:
        pass
