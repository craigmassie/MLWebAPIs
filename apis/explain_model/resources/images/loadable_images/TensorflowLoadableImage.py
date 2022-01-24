from pydantic import BaseModel
from pydantic.typing import Literal

from apis.explain_model.resources.images.images import ModelImage
from apis.explain_model.resources.images.images import ModelImage
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.applications import imagenet_utils
from apis.explain_model.resources.images.loadable_images.ILoadableImage import ILoadableImage


class TensorflowLoadableImage(ILoadableImage):
    type: Literal['Tensorflow']
    preprocess_type = "tf"

    def load(self, input_dimensions) -> ModelImage.ModelImage:
        raise HTTPException(status_code=501, detail=f"Tensorflow models not supported")
