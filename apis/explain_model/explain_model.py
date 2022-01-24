import pathlib

import uvicorn as uvicorn
from fastapi import FastAPI, status

from apis.explain_model.resources.images.images.ModelImage import ModelImage
from apis.explain_model.resources.images.loadable_images import ILoadableImage
from apis.explain_model.resources.images.loadable_images.LoadableImageType import LoadableImageType
from apis.explain_model.resources.models.loadable_models.LoadableModelType import LoadableModelType
from apis.explain_model.resources.models.models import Model
from matplotlib.pyplot import imsave

app = FastAPI(debug=True)


@app.post("/explain-images", status_code=status.HTTP_201_CREATED)
def explain_images(model_location: LoadableModelType, image_location: LoadableImageType, save_location: pathlib.Path):
    """
    Given a request containing a model location and an images location, saves an images with the boundaries of the
    top classification drawn over the images to the path save location.
    """
    model: Model = model_location.model.load()
    input_image: ModelImage = image_location.image.load(model.input_dimensions)

    marked_image, mask = model.explain_image_instance(input_image.image)

    imsave(save_location, marked_image.apply_mask_on_image(mask))


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
