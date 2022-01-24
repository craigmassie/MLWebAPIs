from fastapi import HTTPException
from pydantic import BaseModel
from pydantic.typing import Literal
from apis.explain_model.resources.models.loadable_models.ILoadableModel import ILoadableModel
from apis.explain_model.resources.models.models import KerasModel
import tensorflow as tf


class KerasLoadableModel(ILoadableModel):
    type: Literal['Keras']

    def load(self) -> KerasModel:
        try:
            return KerasModel.KerasModel(tf.keras.models.load_model(self.path))
        except IOError as e:
            raise HTTPException(status_code=404, detail=f"Model could not be loaded from IO, {e}")