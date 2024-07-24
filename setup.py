from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='pca_tools',
    version='0.2.6',
    description='PCA tools for data analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',    
    url='https://github.com/Fidaeic/pca_tools',
    author='Fidae El Morer',
    author_email='elmorer.fidae@gmail.com',
    license='MIT',
    packages=['pca_tools'],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'altair',
        'statsmodels',
        'scipy',
    ],
    test_suite='tests',  # Path to the test suite
    tests_require=[
        'pytest',
    ],
)