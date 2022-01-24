from pydantic.typing import Literal
from apis.explain_model.resources.images.images import ModelImage
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.applications import imagenet_utils
from apis.explain_model.resources.images.loadable_images.ILoadableImage import ILoadableImage
from apis.explain_model.resources.models.models.SupportedLibraryTypes import SupportedLibraryTypes


class KerasLoadableImage(ILoadableImage):
    type: Literal[SupportedLibraryTypes.Keras]

    def load(self, input_dimensions) -> ModelImage.ModelImage:
        return ModelImage.ModelImage(self._normalise(input_dimensions))

    def _normalise(self, input_dimensions):
        pillow_image = image.load_img(self.path, target_size=input_dimensions)
        numpy_image = image.img_to_array(pillow_image)
        return imagenet_utils.preprocess_input(numpy_image, mode="tf")
