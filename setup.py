import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MGwave",
    version="0.1",
    author="Scott Lucchini",
    author_email="lucchini@wisc.edu",
    description="A wavelet transformation package for astronomical data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DOnghiaGroup/MGwave",
    project_urls={
        "Bug Tracker": "https://github.com/DOnghiaGroup/MGwave/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
