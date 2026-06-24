from setuptools import setup, find_packages

setup(
    name="cv-project-template",
    version="0.1.0",
    description="Учебный проект: детекция и классификация объектов дорожной обстановки",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
