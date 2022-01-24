from skimage.segmentation import mark_boundaries


class ModelImage:
    def __init__(self, image):
        self._image = image

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        self._image = image

    def apply_mask_on_image(self, mask):
        return mark_boundaries(self._image / 2 + 0.5, mask)
