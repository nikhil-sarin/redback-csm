from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='redback-csm',
    version='0.1.0',
    description='Fortran-based CSM interaction models for the redback transient inference package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nikhil-sarin/redback-csm',
    author='Nikhil Sarin, Ryosuke Hirai',
    author_email='nsarin.astro@gmail.com',
    license='GNU General Public License v3 (GPLv3)',
    packages=['redback_csm'],
    package_data={'redback_csm': ['priors/*.prior', 'csm*.so', 'csm*.pyd']},
    install_requires=[
        'numpy',
        'scipy',
        'redback>=1.15.0',
    ],
    entry_points={
        'redback.model.modules': [
            'csm_models = redback_csm.models',
        ],
        'redback.model.priors': [
            'csm_priors = redback_csm.prior_provider:get_prior',
        ],
    },
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    zip_safe=False,
)
