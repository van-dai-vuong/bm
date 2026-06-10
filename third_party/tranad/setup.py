from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess
import os
import shutil


def fetch_tranad_src():
    install_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(install_dir, 'src')
    # if submodule was already populated (local dev), skip cloning
    if os.path.exists(src_dir) and os.listdir(src_dir):
        return
    # otherwise fetch it (pip install case)
    tmp = '/tmp/tranad_clone'
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    subprocess.check_call([
        'git', 'clone',
        'https://github.com/imperial-qore/TranAD', tmp
    ])
    shutil.copytree(os.path.join(tmp, 'src'), src_dir)
    shutil.rmtree(tmp)


class PostDevelop(develop):
    def run(self):
        fetch_tranad_src()
        super().run()


class PostInstall(install):
    def run(self):
        fetch_tranad_src()
        super().run()


setup(
    name='tranad',
    version='0.1.0',
    packages=find_packages(),
    cmdclass={
        'develop': PostDevelop,
        'install': PostInstall,
    },
    install_requires=[
        'torch>=1.8.1',
        'torchvision>=0.9.1',
        'torchaudio>=0.8.1',
        'scikit-learn',
        'scipy',
    ],
)