import abc
import pathlib
from pydantic import BaseModel
from apis.explain_model.resources.images.images import ModelImage


class ILoadableImage(BaseModel, abc.ABC):
    path: pathlib.Path

    @abc.abstractmethod
    def load(self, input_dimensions) -> ModelImage:
        pass
