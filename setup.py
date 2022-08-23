from setuptools import setup

setup(
    name='enhope',
    version='0.1.0',    
    description='Enhope',
    url='https://github.com/barbarabenato/enhope.git',
    author='Bárbara C. Benato',
    author_email='barbara.benato@ic.unicamp.br',
    license='BSD 2-clause',
    packages=['enhope'],
    install_requires=[
                      'numpy',
                      'matplotlib',
                      'sklearn', 
                      'torch'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
