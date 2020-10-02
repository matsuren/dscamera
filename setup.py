from os import path

from setuptools import setup

from dscamera import __version__

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dscamera",
    version=__version__,
    license="MIT License",
    install_requires=["numpy", "opencv-python"],
    description="Python library for Double Sphere Camera Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ren Komatsu",
    author_email="komatsu@robot.t.u-tokyo.ac.jp",
    url="https://github.com/matsuren/dscamera",
    packages=["dscamera"],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
