from setuptools import setup

VERSION = '0.1.1'

long_description = '''PyFiberAmp is a set of tools for modeling rare earth fiber amplifiers 
using the Giles rate equation model.'''

setup(
    name='PyFiberAmp',
    version=VERSION,
    author='Joona Rissanen',
    author_email='joona.m.rissanen@gmail.com',
    url='https://github.com/Jomiri/pyfiberamp',
    description='Fiber amplifier modeling library',
    long_description=long_description,
    license='MIT',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy'
    ],
    packages=[
        'pyfiberamp',
        'pyfiberamp.fibers',
        'pyfiberamp.models',
        'pyfiberamp.spectroscopies',
        'pyfiberamp.util',
    ],
    include_package_data=True
)
