from typing import Union

from typing_extensions import Annotated

from apis.explain_model.resources.models.loadable_models.KerasLoadableModel import KerasLoadableModel
from pydantic import BaseModel, Field
from apis.explain_model.resources.models.loadable_models.TensorflowLoadableModel import TensorflowLoadableModel


class LoadableModelType(BaseModel):
    model: Annotated[Union[KerasLoadableModel, TensorflowLoadableModel], Field(discriminator='type')]