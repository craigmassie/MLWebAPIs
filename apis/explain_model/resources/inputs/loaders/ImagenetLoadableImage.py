from pydantic.typing import Literal

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
        pillow_image = image.load_img(self.path, target_size=input_dimensions)
        numpy_image = image.img_to_array(pillow_image)
        return imagenet_utils.preprocess_input(numpy_image, mode="tf")
