import os
from setuptools import setup, find_packages

# Current Version Number
version = "0.1.0"

# Description from README.md
long_description = "\n\n".join([
    open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "README.md"
        ),
        "r"
    ).read()
])

# Required packages
requires = [
    "numpy",
    "pandas",
    "scikit-learn",
    "anndata",
    "tqdm"
]

setup(
    name="bio_grns",
    version=version,
    description="Biological Gene Regulatory Network Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asistradition/biological_grn_simulator",
    author="Chris Jackson",
    author_email="cj59@nyu.edu",
    maintainer="Chris Jackson",
    maintainer_email="cj59@nyu.edu",
    packages=find_packages(
        include=["bio_grns", "bio_grns.*"],
        exclude=["tests", "*.tests"]
    ),
    zip_safe=False,
    install_requires=requires,
    python_requires=">=3.8",
    tests_require=["coverage", "pytest"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta"
    ]
)
