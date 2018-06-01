import os
import io
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


long_description = read("README.md")
version = find_version('image_dataset_viz', '__init__.py')


setup(
    name="image_dataset_viz",
    version=version,
    description=u"Observe dataset of images and targets in few shots",
    long_description=long_description,
    author="vfdev-5",
    author_email="vfdev dot 5 at gmail dot com",
    url="https://github.com/vfdev-5/ImageDatasetViz",
    packages=find_packages(exclude=['tests', 'examples']),
    install_requires=[
        'numpy',
        'Pillow',
        'tqdm',
        'pathlib2;python_version<"3"'
    ],
    license='MIT',
    test_suite="tests",
    extras_require={
        'tests': [
            'pytest',
            'pytest-cov'
        ]
    }
)
