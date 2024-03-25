from setuptools import setup

setup(name='pca_tools',
        version='0.1',
        description='PCA tools for data analysis',
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
            'altair'
        ],
)