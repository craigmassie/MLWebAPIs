from fastapi import HTTPException
from pydantic.typing import Literal
from apis.explain_model.resources.images.images import ModelImage
from apis.explain_model.resources.images.loadable_images.ILoadableImage import ILoadableImage
from apis.explain_model.resources.models.models.SupportedLibraryTypes import SupportedLibraryTypes


class TensorflowLoadableImage(ILoadableImage):
    type: Literal[SupportedLibraryTypes.Tensorflow]

    def load(self, input_dimensions) -> ModelImage.ModelImage:
        raise HTTPException(status_code=501, detail=f"Tensorflow inputs not supported")
