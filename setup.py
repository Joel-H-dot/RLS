from setuptools import setup
import os
import sys

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'RDME.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'RLS - OF',
  packages = ['RLS'],
  version = '1.11',
  license='MIT',
  description = 'Regularised least squares objective function solver',
  long_description_content_type='text/markdown',
  long_description = long_description,
  author = 'Joel Hampton',
  author_email = 'joelelihampton@outlook.com',
  url = 'https://github.com/Joel-H-dot/RLS',
  keywords = ['non_linear optimisation RLS'],
  install_requires=[
          'numpy',
    'TRA',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research ',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)