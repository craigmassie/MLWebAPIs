import pathlib

import uvicorn as uvicorn
from fastapi import FastAPI, status

from apis.explain_model.resources.inputs.images.ModelImage import ModelImage
from apis.explain_model.resources.inputs.loaders.LoadableImageType import LoadableImageType
from apis.explain_model.resources.models.loaders.LoadableModelType import LoadableModelType
from apis.explain_model.resources.models.models import Model
from matplotlib.pyplot import imsave

app = FastAPI(debug=True)


@app.post("/explain-inputs", status_code=status.HTTP_201_CREATED)
def explain_images(model: LoadableModelType,
                   image: LoadableImageType,
                   save_location: pathlib.Path):
    """
    Given a request containing a model location and an inputs location, saves an image with the boundaries of the
    top classification drawn over the inputs to the path save location.
    """
    model: Model = model.model.load()
    input_image: ModelImage = image.image.load(model.input_dimensions)

    explained_image, mask = model.explain_image_instance(input_image.image)

    imsave(save_location, explained_image.apply_mask_on_image(mask))


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
