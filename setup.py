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
# PyPI test: pip install --index-url https://test.pypi.org/simple/ --upgrade envirorg
# PyPI     : pip install --upgrade envirorg
# No PyPI  : pip install -e <local path where "setup.py" (this file) is located>
#
# EXECUTE (example. "gunicorn" must be installed: "pip install gunicorn")
# (IT WORKS WITH ONLY 1 WORKER!!!)
# gunicorn --workers=1 --log-level=debug --timeout=2000 --bind 0.0.0.0:8081 biobarcoding.rest.main:app
#
from setuptools import setup
from pkg_resources import yield_lines
# from distutils.extension import Extension
from Cython.Build import cythonize
# from Cython.Distutils import build_ext

# "enviro" name is taken, so "envirosa" for "ENVIROnment and Social Sustainability Assessment"
package_name = 'envirossa'
version = '0.1'


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


setup(
    name=package_name,
    version=version,
    install_requires=install_reqs,
    packages=['envirorg', 'envirorg.common', 'envirorg.models',
              'biobarcoding.io',
              'biobarcoding.authentication', 'biobarcoding.authorization', 'biobarcoding.services',
              'biobarcoding.rest'],
    include_package_data=True,
    # cmdclass={'build_ext': build_ext},
    # ext_modules=cythonize(["biobarcoding/common/helper_accel.pyx"], language_level="3"),
    url='https://github.com/nextgendem/bcs-backend',
    license='',
    author='rnebot',
    author_email='rnebot@itccanarias.org',
    description='Organism bar-coding system backend'
)
