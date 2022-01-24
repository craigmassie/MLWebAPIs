from fastapi import HTTPException
from pydantic import BaseModel
from pydantic.typing import Literal
from apis.explain_model.resources.models.loadable_models.ILoadableModel import ILoadableModel
from apis.explain_model.resources.models.models import KerasModel
import tensorflow as tf
from apis.explain_model.resources.models.models.Model import Model


class TensorflowLoadableModel(ILoadableModel):
    type: Literal['Tensorflow']

    def load(self) -> Model:
        raise HTTPException(status_code=501, detail=f"Tensorflow models not supported")