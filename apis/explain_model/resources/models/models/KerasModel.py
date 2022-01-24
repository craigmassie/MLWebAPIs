import typing

from typing import Tuple

from pydantic.typing import Literal

from apis.explain_model.resources.inputs.images import ModelImage
from apis.explain_model.resources.models.models.Model import Model
from tensorflow import is_tensor
from lime import lime_image
from numpy import squeeze


class KerasModel(Model):
    def __init__(self, keras_model: typing.Any):
        self._model = keras_model
        self._input_dimensions = self._get_input_dimensions(keras_model)
        self._explainer = lime_image.LimeImageExplainer()

    def _get_input_dimensions(self, keras_model):
        model_input = keras_model.layers[0].output
        # Remove shape from Tensor object context (if in one)
        if is_tensor(model_input):
            if hasattr(model_input, 'shape'):
                model_input = model_input.shape
            else:
                pass
                # app.logger.error("Unable to determine input shape of model.")
                # abort(400)
        # Remove tuple from list context (if in one)
        if isinstance(model_input, list):
            if len(model_input) == 1:
                model_input = model_input[0]
        # app.logger.info(model_input)
        return model_input[1:3]

    @property
    def input_dimensions(self):
        return self._input_dimensions

    @input_dimensions.setter
    def input_dimensions(self, value):
        self._get_input_dimensions(self.model)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def _predict_func(self, images):
        prediction = self.model.predict(images)
        return squeeze(prediction)

    def explain_image_instance(self, image: ModelImage) -> Tuple[ModelImage.ModelImage, typing.Any]:
        explanation = self._explainer.explain_instance(image, self._predict_func, hide_color=0, num_samples=1000)
        image, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        return ModelImage.ModelImage(image), mask

