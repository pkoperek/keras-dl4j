from distutils.core import setup

setup(
    name='kerasdl4j',
    version='0.1',
    description='Use Deeplearning4j as backend for Keras',
    author='Pawel Koperek',
    author_email='pkoperek@gmail.com',
    url='https://github.com/pkoperek/keras-dl4j',
    packages=['kerasdl4j'],
    install_requires=['keras', 'py4j', 'h5py', 'xxhash'],
)
