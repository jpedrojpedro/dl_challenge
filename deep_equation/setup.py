from typing import List

import setuptools


def read_multiline_as_list(file_path: str) -> List[str]:
    with open(file_path) as fh:
        contents = fh.read().split("\n")
        if contents[-1] == "":
            contents.pop()
        return contents


with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = read_multiline_as_list("requirements.txt")

# classifiers = read_multiline_as_list("classifiers.txt")

setuptools.setup(
    name="deep_equation",
    version="0.0.5",
    author="João Pedro Pinheiro",
    author_email="jpinheiro@puc-rio.br",
    description="Deep Equation Challenge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jpedrojpedro/dl_challenge",
    packages=setuptools.find_packages(include=['deep_equation']),
    include_package_data=True,
    # package_dir={"": "src"},
    # classifiers=classifiers,
    # keywords='web api, restful, AI, NLP, retrieval, neural code search',
    entry_points={
        "console_scripts": [
            # '',
        ],
    },
    python_requires=">=3.7, <=3.9.6",
    install_requires=requirements,
    # extras_require={
    #     "full": first_party_deps,
    # },
)
