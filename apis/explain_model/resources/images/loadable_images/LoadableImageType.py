from typing import Union

from typing_extensions import Annotated
from apis.explain_model.resources.images.loadable_images.KerasLoadableImage import KerasLoadableImage
from pydantic import BaseModel, Field
from apis.explain_model.resources.images.loadable_images.TensorflowLoadableImage import TensorflowLoadableImage


class LoadableImageType(BaseModel):
    image: Annotated[Union[KerasLoadableImage, TensorflowLoadableImage], Field(discriminator='type')]