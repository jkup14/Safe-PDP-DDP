try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import setup

config = {
    "name": "safety_embedded_ddp_python",
    "version": "1.0",
    "install_requires": ["numpy", "scipy", "matplotlib"],
}

setup(**config, packages=find_packages())
