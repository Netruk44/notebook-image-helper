from math import ceil, sqrt
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image, make_grid

class GeneratedImages(object):
    def __init__(self, image_tensor, expected_range):
        self.current_range = expected_range

        # Clip image to expected range
        self.image_tensor = image_tensor.clamp(*expected_range)

    def to_range(self, output_range):
        old_min, old_max = self.current_range
        new_min, new_max = output_range

        return GeneratedImages((self.image_tensor - old_min) / (old_max - old_min) * (new_max - new_min) + new_min, output_range)

    def to_dtype(self, dtype):
        return GeneratedImages(self.image_tensor.to(dtype), self.current_range)
    
    def to_tensor(self):
        return self.image_tensor

    def to_numpy(self):
        return self.image_tensor.cpu().numpy()

    def to_pil(self):
        return to_pil_image(make_grid(self.image_tensor, nrow=self._get_nrow()))

    def save(self, path):
        save_image(self.image_tensor, path)

    def show(self):
        from IPython.display import display
        display(self.to_pil())
    
    def _get_nrow(self):
        # Want as square as possible.
        return ceil(sqrt(self.image_tensor.shape[0]))