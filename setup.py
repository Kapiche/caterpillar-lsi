from distutils.core import setup
from setuptools.command.test import test as TestCommand
import sys


class Tox(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import tox
        errno = tox.cmdline(self.test_args)
        sys.exit(errno)


requires = [
    'caterpillar>=1.0.0alpha',
    'ujson',
    'numpy',
    'scipy'
]

setup(
    name='caterpillar_lsi',
    version='1.0.0alpha',
    packages=[
        'caterpillar_lsi',
    ],
    url='http://www.kapiche.com',
    install_requires=requires,
    tests_require=['tox', 'pytest', 'coverage', 'pep8'],
    cmdclass={'test': Tox},
    author='Kapiche,',
    author_email='contact@kapiche.com',
    description='LSI plugin for caterpillar text analytics engine.'
)
