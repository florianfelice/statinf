import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="{library}",
    version="{version}",
    author="Florian Felice",
    author_email="florian.felice@outlook.com",
    description="{desc}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/florianfelice/{library}",
    packages=setuptools.find_packages(),
    install_requires=[
          {requirements}
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)