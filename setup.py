from setuptools import setup, find_packages

setup(
    name='myMLpackage',
    version='1.0.0',
    packages=find_packages(),
    package_data={'myMLpackage': ['data/*.csv', 'data/*.json']},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'pycaret',
        'streamlit',
        # Add any other dependencies here
    ],
)
