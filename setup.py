import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join( os.path.dirname(__file__), fname) ).read()


setup(
    name='SLICE',
    version='0.0.1',
    author='George I Austin', 
    author_email='gia2105@columbia.edu', 
    url='https://github.com/korem-lab/SLICE',
    packages=['slice'],
    include_package_data=True,
    install_requires=['numpy', 
                      'pandas',
                      'scikit-learn' 
                     ]
)
