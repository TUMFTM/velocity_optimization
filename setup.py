import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="velocity_optimization",
    version="0.4",
    author="Thomas Herrmann",
    author_email="thomas.herrmann@tum.de",
    description="Optimizes (Maximizes) the velocity profile for a vehicle respecting physical constraints and runtime-variable input parameters.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TUMFTM/velocity_optimization",
    packages=setuptools.find_packages(),  # TODO: only necessary packages to run code on vehicle
    include_package_data=True,
    install_requires=[line.strip() for line in open("requirements.txt").readlines()],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
