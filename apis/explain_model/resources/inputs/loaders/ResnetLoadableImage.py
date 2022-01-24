from fastapi import HTTPException
from pydantic.typing import Literal

from apis.explain_model.resources.inputs.SupportedPreprocessorTypes import SupportedPreprocessorTypes
from apis.explain_model.resources.inputs.images import ModelImage
from apis.explain_model.resources.inputs.loaders.ILoadableImage import ILoadableImage


class ResnetLoadableImage(ILoadableImage):
    type: Literal[SupportedPreprocessorTypes.ResNet]

    def load(self, input_dimensions) -> ModelImage.ModelImage:
        raise HTTPException(status_code=501, detail=f"Tensorflow inputs not supported")
