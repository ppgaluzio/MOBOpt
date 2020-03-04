import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MOBOpt",
    version="1.0.0",
    author="P. P. Galuzio",
    author_email="galuzio.paulo@protonmail.com",
    description="Multi-Objective Bayesian Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ppgaluzio/PastaPhase",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
