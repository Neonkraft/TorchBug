from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requires = f.read().splitlines()

setup(
    name='PyCaliper',
    description='A lightweight framework to compare different PyTorch implementations of the same model',
    author='Arjun Krishnakumar',
    author_email='arjun.krishnakumar@students.uni-freiburg.de',
    version='0.1dev',
    install_requires=requires,
    packages=find_packages(),
    license='Apache License 2.0',
)