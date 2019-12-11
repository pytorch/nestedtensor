import setuptools
import datetime
import torch
import distutils.command.clean
import shutil
import os
import glob
import subprocess
import sys

from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME, BuildExtension


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


latest_release = "0.0.1"

dt = datetime.datetime.utcnow()
package_version = "{0}.dev{1}{2}{3}{4}".format(
    latest_release, dt.year, dt.month, dt.day, dt.hour)

sha = 'Unknown'
package_name = 'nestedtensor'

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    version = package_version + '+' + sha[:7]
else:
    version = package_version
print("Building wheel {}-{}".format(package_name, version))


def write_version_file():
    version_path = os.path.join(cwd, 'nestedtensor', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))
        f.write("from nestedtensor import _C\n")
        f.write("if hasattr(_C, 'CUDA_VERSION'):\n")
        f.write("    cuda = _C.CUDA_VERSION\n")


write_version_file()

readme = open('README.md').read()

pytorch_dep = 'torch'

requirements = [
    pytorch_dep,
]

if os.getenv('PYTORCH_VERSION'):
    pytorch_dep += "==" + os.getenv('PYTORCH_VERSION')


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'nestedtensor', 'csrc')

    sources = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    extension = CppExtension

    define_macros = []

    extra_compile_args = {'cxx': ['-O0', '-g']}
    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv('FORCE_CUDA', '0') == '1':
        extension = CUDAExtension
        define_macros += [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        if nvcc_flags == '':
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(' ')
        extra_compile_args['nvcc'] = nvcc_flags

    if sys.platform == 'win32':
        define_macros += [('nestedtensor_EXPORTS', None)]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            'nestedtensor._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


class clean(distutils.command.clean.clean):
    def run(self):
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split('\n')):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


setuptools.setup(
    name=package_name,
    version=package_version,
    author="Christian Puhrsch",
    author_email="cpuhrsch@fb.com",
    description="NestedTensors for PyTorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch/nestedtensor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=True,
    cmdclass={'clean': clean, 'build_ext': BuildExtension},
    install_requires=requirements,
    ext_modules=get_extensions()
)
