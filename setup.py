# ONE TIME  -----------------------------------------------------------------------------
#
# pip install --upgrade setuptools wheel twine
#
# Create account:
# PyPI test: https://test.pypi.org/account/register/
# or PyPI  : https://pypi.org/account/register/
#
# EACH TIME -----------------------------------------------------------------------------
#
# Modify version code in "setup.py" (this file)
#
# Build (cd to directory where "setup.py" is)
# python3 setup.py sdist bdist_wheel
#
# Upload:
# PyPI test: twine upload --skip-existing --repository-url https://test.pypi.org/legacy/ dist/*
# or PyPI  : twine upload --skip-existing dist/*
#
# INSTALL   ------------------------------------------------------------------------------
#
# Local    : python setup.py install
# PyPI test: pip install --index-url https://test.pypi.org/simple/ --upgrade enbios
# PyPI     : pip install --upgrade enbios
# No PyPI  : pip install -e <local path where "setup.py" (this file) is located>
#
from os import path
from setuptools import setup
from pkg_resources import yield_lines
# from distutils.extension import Extension
# from Cython.Build import cythonize
# from Cython.Distutils import build_ext

"""
python3 setup.py sdist bdist_wheel
twine upload --skip-existing dist/*
"""

package_name = 'enbios'
version = '0.20'


def parse_requirements(strs):
    """Yield ``Requirement`` objects for each specification in `strs`

    `strs` must be a string, or a (possibly-nested) iterable thereof.
    """
    # create a steppable iterator, so we can handle \-continuations
    lines = iter(yield_lines(strs))

    ret = []
    for line in lines:
        # Drop comments -- a hash without a space may be in a URL.
        if ' #' in line:
            line = line[:line.find(' #')]
        # If there is a line continuation, drop it, and append the next line.
        if line.endswith('\\'):
            line = line[:-2].strip()
            try:
                line += next(lines)
            except StopIteration:
                return
        ret.append(line)

    return ret


with open('requirements.txt') as f:
    required = f.read().splitlines()

install_reqs = parse_requirements(required)
print(install_reqs)

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=package_name,
    version=version,
    install_requires=install_reqs,
    packages=['enbios', 'enbios.bin', 'enbios.common', 'enbios.model',
              'enbios.input', 'enbios.input.data_preparation', 'enbios.input.simulators',
              'enbios.output',
              'enbios.processing', 'enbios.processing.indicators'],
    include_package_data=True,
    url='https://github.com/ENVIRO-Module/enviro',
    license='MIT',
    author=['Rafael Nebot', 'Cristina Madrid'],
    author_email='rnebot@itccanarias.org',
    entry_points={'console_scripts': ['enbios=enbios.bin.script:main']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Indicators of environmental sustainability of energy systems using MuSIASEM and LCA methodologies'
)
