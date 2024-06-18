from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, "r") as file:
        lines = file.read().splitlines()
    # Exclude empty lines and comments
    return [line for line in lines if line and not line.startswith("#")]


setup(
    name="burst_clustering",
    version="0.1",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.11, <4",
)
