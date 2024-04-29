from setuptools import setup, find_packages

setup(
    name='myMLpackage',
    version='1.0.0',
    packages=find_packages(),
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
