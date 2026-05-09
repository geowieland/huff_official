from setuptools import setup, find_packages
import os

def read_README():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
        return f.read()
    
setup(
    name='huff',
    version='1.9.0',
    description='huff: Market Area Analysis in Python',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    long_description=read_README(),
    long_description_content_type='text/markdown',
    author='Thomas Wieland',
    author_email='geowieland@googlemail.com',
    license_files=["LICENSE"],
    package_data={
        'huff': ['tests/data/*'],
    },
    install_requires=[
        'geopandas>=0.14,<0.15',
        'pandas>=2.0,<2.3',
        'numpy>=1.26,<2.0',
        'statsmodels==0.14.2',
        'scipy==1.15.3',
        'scikit-learn>=1.3,<1.6',
        'xgboost>=3.1.0,<=3.2.0',
        'lightgbm==4.6.0',
        'shapely>=2.0,<2.1',
        'requests>=2.31,<3.0',
        'matplotlib>=3.8,<3.9',
        'pillow>=10,<11',
        'contextily>=1.4,<1.6',
        'openpyxl>=3.1,<3.2'
    ],
    test_suite='huff.tests',
)