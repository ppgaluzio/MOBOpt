import setuptools

import re

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)

with open("README.md", "r") as fh:
    long_description = fh.read()

project_name = "mobopt"

setuptools.setup(
    name="MOBOpt",
    version= get_property('__version__', project_name),
    author="P. P. Galuzio",
    author_email="galuzio.paulo@protonmail.com",
    description="Multi-Objective Bayesian Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ppgaluzio/MOBOpt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
