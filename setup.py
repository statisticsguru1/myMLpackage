from setuptools import setup, find_packages

setup(
    name='myMLpackage',
    version='1.0.0',
    packages=find_packages(),
    package_data={'your_package_name': ['data/*.csv', 'data/*.json']}
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
