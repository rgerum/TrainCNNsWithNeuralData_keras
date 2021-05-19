from setuptools import setup

setup(name='brain_regularizer',
      version="0.1",
      packages=['brain_regularizer'],
      description='Train a neural network with brain data as a regularizer',
      author='Richard Gerum',
      author_email='richard.gerum@fau.de',
      license='MIT',
      install_requires=[
          'numpy',
          'scipy',
          'tensorflow',
          'pandas',
      ],
)
