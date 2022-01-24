from typing import Union
from typing_extensions import Annotated
from apis.explain_model.resources.inputs.loaders.ImagenetLoadableImage import ImagenetLoadableImage
from pydantic import BaseModel, Field
from apis.explain_model.resources.inputs.loaders.ResnetLoadableImage import ResnetLoadableImage


class LoadableImageType(BaseModel):
    image: Annotated[Union[ImagenetLoadableImage, ResnetLoadableImage], Field(discriminator='type')]
