# from setuptools import setup, find_packages
#
# setup(
#     name='highway-env',
#     version='1.10.0',
#     packages=find_packages(),
#     description='A simple Python package',
#     author='LYZ_change',
#     # author_email='your_email@example.com',
#     # url='https://github.com/yourusername/your_package',
#     install_requires=[
#         # 列出你的项目依赖的其他包
#         # 例如: 'numpy', 'pandas'
#     ],
# )
# Following PEP 517/518, this file should not not needed and replaced instead by the setup.cfg file and pyproject.toml.
# Unfortunately it is still required py the pip editable mode `pip install -e`
# See https://stackoverflow.com/a/60885212

import pathlib
from setuptools import setup

# 直接指定版本号
VERSION = '1.0.0'

CWD = pathlib.Path(__file__).absolute().parent

setup(version=VERSION)