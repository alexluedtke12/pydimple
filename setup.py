from setuptools import setup, find_packages

setup(
    name='pydimple',
    version='0.1',
    author='Alex Luedtke',
    author_email='aluedtke@uw.edu',
    license='MIT',
    description='Proof-of-concept implementation of dimple (debiased inference made simple)',
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'optuna==3.2.0', # requiring specific versions of optuna and lightgbm because lightgbm removed fobj argument for custom objectives, and I'm unclear on whether Optuna has been updated to account for this
        'lightgbm==3.3.5',
        'sympy',
        'statsmodels'
    ],
    python_requires='>=3.8',
)
