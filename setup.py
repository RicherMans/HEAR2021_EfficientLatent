import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='efficient_latent',
    version='0.0.1',
    description='EfficientNet embeddings for HEAR',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/richermans/HEAR2021_EfficientLatent',
    packages=setuptools.find_packages(),
    author='Heinrich Dinkel',
    author_email='heinrich.dinkel@gmail.com',
    license='Apache License 2.0',
    install_requires=[
        "librosa",
        "torch",
        "torchaudio",
        "numpy==1.19.5",
        "einops",
        "efficientnet_pytorch==0.7.1",
        "numba==0.48",
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)
