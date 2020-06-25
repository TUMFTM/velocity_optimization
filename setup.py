import setuptools

with open("Readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="velocity-planner-mpSQP-Thomas-Herrmann", # Replace with your own username
    version="0.1",
    author="Thomas Herrmann",
    author_email="thomas.herrmann@tum.de",
    description="Optimizes (Maximizes) the velocity profile for a vehicle respecting physical constraints and runtime-variable input parameters.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
