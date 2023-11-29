from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    install_requires=[
        'mlflow==1.20.2',
        'matplotlib==3.4.3',
        'numpy==1.21.3',
        'einops==0.4.6',
        'torch==1.10.0',
        'torchsummary==1.5.1',
        'tqdm==4.62.3',
        'torchvision==0.11.1',
    ],
    version='0.1.0',
    description='Reproducing BiRT Architecture (Bio-inspired replay for transformers in Continual Learning)',
    author='Disha Maheshwari',
    license='',
)
