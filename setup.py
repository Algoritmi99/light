from pathlib import Path

from setuptools import setup, find_packages


# Function to read the requirements.txt file
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


# Safely read the README.md file
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


def version():
    with open(Path(__file__).parent / 'version', 'r') as file:
        v = file.readline()
    return v


setup(
    name="light",
    version=version(),
    description="Lightweight Neural Network and Machine Learning Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Arash T. Goodarzi",
    author_email="arash.orca99@gmail.com",
    url="",
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    include_package_data=True,
)
