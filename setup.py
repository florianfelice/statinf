import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="statspy",
    version="1.0.1",
    author="Florian Felice",
    author_email="florian.felice@outlook.com",
    description="A library for rigorous statistics and econometrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/florianfelice/statspy",
    packages=setuptools.find_packages(),
    install_requires=[
          "pandas>=0.24.1",
          "numpy>=1.16.3",
          "scipy>=1.2.1"
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)