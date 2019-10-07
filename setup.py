import setuptools

REQUIRED_PACKAGE = [
    'tensorflow-probability>=0.8.0'
]

setuptools.setup(
    name='extended_tfp',
    project_name="extended-tensorflow-probability",
    version="0.0.1",
    author="Seungjin Yang",
    author_email="seungjin.yang@cern.ch",
    description="It just has codes based on tensorflow-probability",
    url="https://github.com/seungjin-yang/bayesian-deep-learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=REQUIRED_PACKAGE
)
