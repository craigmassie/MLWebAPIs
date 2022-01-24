from fastapi import HTTPException, status
from pydantic.typing import Literal
from apis.explain_model.resources.models.loaders.ILoadableModel import ILoadableModel
from apis.explain_model.resources.models.models.Model import Model
from apis.explain_model.resources.models.SupportedLibraryTypes import SupportedLibraryTypes


class TensorflowLoadableModel(ILoadableModel):
    type: Literal[SupportedLibraryTypes.Tensorflow]

    def load(self) -> Model:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Tensorflow models not supported")