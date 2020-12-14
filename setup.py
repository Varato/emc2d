import os
import sys
import platform
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


# copied from https://github.com/pybind/cmake_example/blob/master/setup.py
class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        build_dir = os.path.abspath(self.build_temp)
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        cmake_args = [
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        if platform.system() == "Windows":
            cmake_args += ["-GNinja"]

        # Assuming Makefiles
        build_args += ['--', '-j2']

        env = os.environ.copy()
        # env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
        #     env.get('CXXFLAGS', ''),
        #     self.distribution.get_version())

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print('-'*10, 'CMake Configuring', '-'*40)
        subprocess.check_call(['cmake', cmake_list_dir] + cmake_args,
                              cwd=self.build_temp, env=env)

        print('-'*10, 'Building extensions', '-'*40)
        cmake_cmd = ['cmake', '--build', '.'] + build_args
        subprocess.check_call(cmake_cmd, cwd=self.build_temp)

        # Move from build temp to final position
        print('-'*10, 'Moving extensions to right positions', '-'*40)
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext):
        build_temp = Path(self.build_temp).resolve()
        dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        source_path = build_temp / self.get_ext_filename(ext.name)
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        if os.path.isfile(source_path):
            self.copy_file(source_path, dest_path)


ext_modules = [CMakeExtension('emc2d.extensions.emc_kernel')]

setup(
    name='emc2d',
    version='0.1.1',
    author='Varato',
    author_email='imxinchen@outlook.com',
    description='EMC motion correction',
    long_description='',
    packages=find_packages(where='.'),
    # package_data=package_data,
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
