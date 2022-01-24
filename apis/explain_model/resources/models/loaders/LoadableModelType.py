from typing import Union
from typing_extensions import Annotated
from apis.explain_model.resources.models.loaders.KerasLoadableModel import KerasLoadableModel
from pydantic import BaseModel, Field
from apis.explain_model.resources.models.loaders.TensorflowLoadableModel import TensorflowLoadableModel


class LoadableModelType(BaseModel):
    model: Annotated[Union[KerasLoadableModel, TensorflowLoadableModel], Field(discriminator='type')]