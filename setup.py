# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 00:30:16 2025

@author: SSC
"""

from setuptools import setup, find_packages

setup(
    name='darts-gt',
    version='1.0.0',
    author='Shruti Sarika Chakraborty',
    description='DARTS-GT: Differentiable Architecture Search for Graph Transformers',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    python_requires='>=3.8',
)