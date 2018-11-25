from setuptools import setup

NAME = 'ntsa'
DESCRIPTION = 'Neural Time-series Analysis'
URL = 'https://github.com/d3sm0/ntsa'
EMAIL = 'me@d3sm0.com'
AUTHOR = 'd3sm0'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = 0.1

REQUIRED = ['numpy', 'pandas', 'tensorflow', 'gin-config']
EXTRAS = ['jupyter', 'ipython', 'scipy']

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT')
