import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="velocity_optimization",
    version="0.4.3",
    author="Thomas Herrmann",
    author_email="thomas.herrmann@tum.de",
    description="Optimizes (Maximizes) the velocity profile for a vehicle respecting physical constraints and "
                "runtime-variable input parameters.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TUMFTM/velocity_optimization",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["numpy>=1.18.1",  # packages to run code on car
                      "osqp>=0.6.1",
                      "scipy>=1.3.3",
                      "matplotlib>=3.0.3",
                      "sympy>=1.6"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
