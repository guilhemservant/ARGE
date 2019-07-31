import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ARGE",
    version="0.0.1",
    author="Guilhem Servant",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['pandas>=0.21.0', 'numpy'],
    entry_points={'console_scripts': ['ARGE=ARGE.ARGE:entry_point']}
)