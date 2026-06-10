from setuptools import setup, find_packages

setup(
    name='tranad',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.1',
        'torchvision>=0.9.1',
        'torchaudio>=0.8.1',
        'scikit-learn',
        'tqdm',
        'scipy',
    ],
)
