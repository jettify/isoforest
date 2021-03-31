import os
import re

from setuptools import find_packages, setup
from setuptools_rust import Binding, RustExtension

install_requires = []


def _read(f):
    with open(os.path.join(os.path.dirname(__file__), f)) as f_:
        return f_.read().strip()


def _read_version():
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(
        os.path.dirname(__file__), 'isoforest', '__init__.py'
    )
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        raise RuntimeError('Cannot find version in isoforest/__init__.py')


classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Operating System :: OS Independent',
    'Development Status :: 3 - Alpha',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

keywords = ['isolation forest', 'linfa']

project_urls = {
    'Website': 'https://github.com/jettify/isoforest',
    'Documentation': 'https://isoforest.readthedocs.io',
    'Issues': 'https://github.com/jettify/isoforest/issues',
}


setup(
    name='isoforest',
    version=_read_version(),
    description=('isoforest'),
    long_description='\n\n'.join((_read('README.rst'), _read('CHANGES.rst'))),
    long_description_content_type='text/x-rst',
    classifiers=classifiers,
    platforms=['POSIX'],
    author='Nikolay Novik',
    author_email='nickolainovik@gmail.com',
    url='https://github.com/jettify/isoforest',
    download_url='https://pypi.org/project/isoforest/',
    license='Apache 2',
    packages=['isoforest'],
    install_requires=install_requires,
    keywords=keywords,
    include_package_data=True,
    project_urls=project_urls,
    python_requires='>=3.6.0',
    rust_extensions=[
        RustExtension("isoforest._isoforest", binding=Binding.PyO3)
    ],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
