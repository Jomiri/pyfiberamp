from setuptools import setup

VERSION = '0.2.0dev'

long_description = '''PyFiberAmp is a set of tools for modeling rare-earth-doped fiber laser and amplifiers 
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
        'pyfiberamp.dynamic',
        'pyfiberamp.steady_state',
        'pyfiberamp.spectroscopies',
        'pyfiberamp.util',
    ],
    include_package_data=True,
    package_data={'pyfiberamp': ['spectroscopies/fiber_spectra/*.dat']}
)
