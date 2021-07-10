from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=6", "nbformat>=4", "nbconvert>=5", "requests>=2"]

setup(
    name="bi-connect",
    version="0.0.1",
    author="Cat Vasily",
    author_email="olbersia@gmail.com",
    description="A package to compute connectivity between signals",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/catvasily/pysynchrony",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
    ],
)