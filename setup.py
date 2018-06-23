#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree
from setuptools import setup, find_packages, Command

NAME = 'riconv'
DESCRIPTION = 'A convolution layer incorporating rotational invariance',
MAINTAINER = 'Zigfried Hampel-Arias'
MAINTAINER_EMAIL = 'zhampel@gmail.com'
URL = 'https://github.com/zhampel/rot-inv-conv'
LICENSE = 'MIT'


here = os.path.abspath(os.path.dirname(__file__))

def read(path, encoding='utf-8'):
    with io.open(path, encoding=encoding) as f:
        content = f.read()
    return content

def get_install_requirements(path):
    content = read(path)
    requirements = [req for req in content.split("\n")
                    if req != '' and not req.startswith('#')]
    return requirements

# README
LONG_DESCRIPTION = read(os.path.join(here, 'README.md'))


# Want to read in package version number from __version__.py
about = {}
with io.open(os.path.join(here, 'riconv', '__version__.py'), encoding='utf-8') as f:
    exec(f.read(), about)
    VERSION = about['__version__']

# requirements
INSTALL_REQUIRES = get_install_requirements(os.path.join(here, 'requirements.txt'))

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    license=LICENSE,
    long_description=LONG_DESCRIPTION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    url=URL,
    packages=['riconv'],
    #packages=find_packages(),
    install_requires=INSTALL_REQUIRES, #external packages as dependencies
    setup_requires=['setuptools>=38.6.0'],
    scripts=[
             'scripts/train.py',
             'scripts/test_all.py',
             'scripts/test_batch.py',
             'scripts/aug_examples.py',
             'scripts/make_training_figures.py',
            ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
