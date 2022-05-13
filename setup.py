from setuptools import setup

setup(
    name="notebook-image-helper",
    py_modules=["image_helper"],
    install_requires=["torch", "torchvision"],
)