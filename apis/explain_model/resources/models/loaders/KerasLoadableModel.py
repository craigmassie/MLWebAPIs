from fastapi import HTTPException, status
from pydantic.typing import Literal
from apis.explain_model.resources.models.loaders.ILoadableModel import ILoadableModel
from apis.explain_model.resources.models.models import KerasModel
import tensorflow as tf
from apis.explain_model.resources.models.SupportedLibraryTypes import SupportedLibraryTypes


class KerasLoadableModel(ILoadableModel):
    type: Literal[SupportedLibraryTypes.Keras]

    def load(self) -> KerasModel:
        try:
            return KerasModel.KerasModel(tf.keras.models.load_model(self.path))
        except IOError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model could not be loaded from IO, {e}")