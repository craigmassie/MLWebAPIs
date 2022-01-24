import abc
import pathlib

from pydantic import BaseModel
from apis.explain_model.resources.models.models import Model


class ILoadableModel(BaseModel, abc.ABC):
    path: pathlib.Path

    @abc.abstractmethod
    def load(self) -> Model:
        pass