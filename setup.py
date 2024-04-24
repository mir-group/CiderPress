import os
import sys

from numpy.distutils.command.build import build
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.core import Extension as NPExtension
from numpy.distutils.core import setup
from setuptools import find_packages

fext_dir = "ciderpress/lib/gpaw_utils_src"
fsources = ["gpaw_utils.f90", "fast_sph_harm.f90"]
fext = NPExtension(
    name="ciderpress.dft.futil",
    sources=[os.path.join(fext_dir, fsrc) for fsrc in fsources],
    f2py_options=["--quiet"],
)


class CMakeBuildExt(build_ext):
    def run(self):
        super(CMakeBuildExt, self).run()
        self.build_cmake(None)

    def build_cmake(self, extension):
        self.announce("Configuring extensions", level=3)
        src_dir = os.path.abspath(os.path.join(__file__, "..", "ciderpress", "lib"))
        cmd = [
            "cmake",
            f"-S{src_dir}",
            f"-B{self.build_temp}",
            "-DCMAKE_PREFIX_PATH={}".format(sys.base_prefix),
            "-DBLA_VENDOR=Intel10_64lp_seq",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        configure_args = os.getenv("CMAKE_CONFIGURE_ARGS")
        if configure_args:
            cmd.extend(configure_args.split(" "))
        self.spawn(cmd)

        self.announce("Building binaries", level=3)
        cmd = ["cmake", "--build", self.build_temp, "-j2"]
        build_args = os.getenv("CMAKE_BUILD_ARGS")
        if build_args:
            cmd.extend(build_args.split(" "))
        if self.dry_run:
            self.announce(" ".join(cmd))
        else:
            self.spawn(cmd)

    def get_ext_filename(self, ext_name):
        if "ciderpress.lib" in ext_name:
            ext_path = ext_name.split(".")
            filename = build_ext.get_ext_filename(self, ext_name)
            name, ext_suffix = os.path.splitext(filename)
            return os.path.join(*ext_path) + ext_suffix
        else:
            return super(CMakeBuildExt, self).get_ext_filename(ext_name)


build.sub_commands = [c for c in build.sub_commands if c[0] == "build_ext"] + [
    c for c in build.sub_commands if c[0] != "build_ext"
]

# TODO: need to add gpaw>=22.8.1b1 to reqs at some point
with open("requirements.txt", "r") as f:
    requirements = [l.strip() for l in f.readlines()]

description = """CiderPress is a package for running DFT calculations
with CIDER functionals in the GPAW and PySCF codes."""

setup(
    name="ciderpress",
    description=description,
    version="0.0.10",
    packages=find_packages(exclude=["*test*", "*examples*"]),
    ext_modules=[fext],
    cmdclass={"build_ext": CMakeBuildExt},
    setup_requires=["numpy"],
    install_requires=requirements,
)
