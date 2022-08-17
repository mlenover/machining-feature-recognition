# flake8:noqa=W191
from __future__ import unicode_literals
files = {}
dirs  = set()

files['setup.py'] = setup_py = """
#!/usr/bin/env python
from __future__ import absolute_import
from setuptools import setup
from subprocess import check_output
import json

def setupargs():
    try:                import makepy.config as m; return m.read_setup_args()
    except ImportError: return json.loads(check_output(['makepy', 'setupargs']).decode('utf-8'))

if __name__ == '__main__': setup(**setupargs())
"""

files['.gitignore'] = _gitignore = """
.cache
build
dist
*.egg-info
.asc
transpiled
backport
.pytest_cache
.tox
__pycache__
*.pyc
*.pyo
*.pyd
*.py-e
.vol
"""
