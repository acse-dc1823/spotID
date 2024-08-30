# author: David Colomer Matachana
# GitHub username: acse-dc1823

import torchvision.transforms as T


class ResizeTransform:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.transform = T.Resize((height, width),
                                  interpolation=T.InterpolationMode.BILINEAR)

    def __call__(self, image):
        return self.transform(image)
