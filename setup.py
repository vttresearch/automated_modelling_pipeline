from setuptools import find_packages, setup


# Function to read extras_require from a file
def read_extras_requirements(filename='extras_requirements.txt'):
    extras_require = {}
    with open(filename, 'r') as f:
        current_feature = None
        for line in f:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_feature = line[1:-1]
                extras_require[current_feature] = []
            elif current_feature and line:
                extras_require[current_feature].append(line)
    return extras_require


def read_requirements(path):
    with open(path) as f:
        return f.read().splitlines()


base_reqs = read_requirements("requirements.txt")
reqs = base_reqs

with open("README.md") as fh:
    LONG_DESCRIPTION = fh.read()


URL = "https://github.com/vttresearch/automated_modelling_pipeline"


PROJECT_URLS = {
    "Bug Tracker": "https://github.com/vttresearch/automated_modelling_pipeline/issues",
    "Documentation": URL,
    "Source Code": "https://github.com/vttresearch/automated_modelling_pipeline",
}


setup(
    name="amp",
    version="0.1",
    description="Automatic Modelling Pipeline:"
                "A tool aimed especially for creating forecasting models of the building energy systems",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    project_urls=PROJECT_URLS,
    url=URL,
    maintainer="VTT",
    license="LicenseRef-AMP-Non-Commercial",
    license_files=("LICENSE",),
    classifiers=[
        "License :: Other/Proprietary License",
    ],
    packages=find_packages(),
    install_requires=reqs,
    extras_require=read_extras_requirements('extras_requirements.txt'),
    python_requires=">=3.11.13",

)