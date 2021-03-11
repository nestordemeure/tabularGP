from setuptools import setup, find_packages

setup(
  name = 'tabularGP',
  version = '1.0',
  license='apache-2.0',
  description = 'Use gaussian processes on tabular data as a drop-in replacement for neural networks',
  author = 'NestorDemeure',
#  author_email = 'your.email@domain.com',
  url = 'https://github.com/nestordemeure/tabularGP',
  download_url = 'https://github.com/nestordemeure/tabularGP/archive/v1.1.tar.gz',
  keywords = ['gaussian-processes', 'tabular-data', 'deep-learning', 'pytorch', 'fastai'],
  install_requires=[
          'numpy',
          'pandas',
          'torch',
          'fastai',
      ],
  classifiers=[ # https://pypi.org/classifiers/
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3 :: Only',
  ],
  packages=find_packages(),
)
