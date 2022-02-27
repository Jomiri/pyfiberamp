from setuptools import setup, dist

try:
    # Try to compile pythran binary extension module.
    from pythran.dist import PythranExtension, PythranBuildExt
    dist.Distribution(dict(setup_requires='pythran'))
    ext_modules = [
        PythranExtension(
            'pyfiberamp.dynamic.fiber_simulation_pythran_bindings',
            ['pyfiberamp/dynamic/inner_loop_functions.py'],
        ),
    ]
    cmdclass = {"build_ext": PythranBuildExt}
except Exception:
    ext_modules = []
    cmdclass = {}

VERSION = '0.5.0'

long_description = '''PyFiberAmp is a powerful simulation library for modeling
rare-earth-doped fiber lasers and amplifiers using rate equations.'''

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
        'pyfiberamp.steady_state.models',
        'pyfiberamp.spectroscopies',
        'pyfiberamp.util',
        'pyfiberamp.mode_solver',
        'pyfiberamp.plotting'
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    package_data={'pyfiberamp': ['spectroscopies/fiber_spectra/*.dat',
                                 'dynamic/fiber_simulation_pybindings.cp39-win_amd64.pyd']}
)
