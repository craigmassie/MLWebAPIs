from pydantic.typing import Literal
from fastapi import HTTPException, status
from apis.explain_model.resources.inputs.SupportedPreprocessorTypes import SupportedPreprocessorTypes
from apis.explain_model.resources.inputs.images import ModelImage
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.applications import imagenet_utils
from apis.explain_model.resources.inputs.loaders.ILoadableImage import ILoadableImage


class ImagenetLoadableImage(ILoadableImage):
    type: Literal[SupportedPreprocessorTypes.ImageNet]

    def load(self, input_dimensions) -> ModelImage.ModelImage:
        return ModelImage.ModelImage(self._normalise(input_dimensions))

    def _normalise(self, input_dimensions):
        try:
            pillow_image = image.load_img(self.path, target_size=input_dimensions)
        except FileNotFoundError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image could not be loaded from IO, {e}")
        numpy_image = image.img_to_array(pillow_image)
        return imagenet_utils.preprocess_input(numpy_image, mode="tf")
