# flake8:noqa=W191
from __future__ import unicode_literals
files = {}
dirs  = set()

dirs.add('make')
files['make/vars.mk'] = make_vars_mk = """
# Generic Python Project Make-Include File
# ========================================
# Copy these mk-files to your python project for easy testing, building, and publishing.
#
# (c) Uwe Jugel, @ubunatic, License: MIT
#
# Integration and Usage
# ---------------------
# Simply `include make/makepy.mk` to include all mk files in your `Makefile`.
# Now you can use `make vars`, `make pyclean`, and `make dev-install`.

# This mk file defines the most common vars for building Python projects.
# Run `make PY=2` or `make PY=3` to setup vars for either Python 2 or 3.

# Using PYTHONPATH leads to unpredicted behavior.
unexport PYTHONPATH

# The default project name is the name of the current dir. All code usually
# resides in another subdir (package) with the same name as the project.
PKG  = $(notdir $(basename $(CURDIR)))
MAIN = $(PKG)

# main python vars, defining python and pip binaries
_GET_MAJOR = 'import sys; sys.stdout.write(str(sys.version_info.major) + "\\n")'
_GET_MINOR = 'import sys; sys.stdout.write(str(sys.version_info.minor) + "\\n")'
PY     := $(shell python -c $(_GET_MAJOR)).$(shell python -c $(_GET_MINOR))
PYTHON = python$(PY)
PIP    = $(PYTHON) -m pip
MAKEPY = $(PYTHON) -m makepy

# export and define setup vars, used for dist building
export WHEELTAG := py$(shell $(PYTHON) -c $(_GET_MAJOR))
ifeq ($(WHEELTAG), py2)
SETUP_DIR = backport
else
SETUP_DIR = $(CURDIR)
endif

# The default tests dir is 'tests'.
# Use PRJ_TESTS = other1 other2 in your Makefile to override.
PRJ_TESTS = $(wildcard ./tests)
# We use regard project files as source files to trigger rebuilds, etc.
PRJ_FILES = tox.ini setup.py setup.cfg Makefile LICENSE.txt README.md
SRC_FILES = $(PKG) $(PRJ_TESTS) $(PRJ_FILES)

# utils and help vars
NOL       = 1>/dev/null  # mute stdout
NEL       = 2>/dev/null  # mute stderr

"""

dirs.add('make')
files['make/tests.mk'] = make_tests_mk = """
# Generic Python Project Make-Include File
# ========================================
# Copy these mk-files to your python project for easy testing, building, and publishing.
#
# (c) Uwe Jugel, @ubunatic, License: MIT
#
# Integration and Usage
# ---------------------
# Simply `include make/makepy.mk` to include all mk files in your `Makefile`.
# Now you can use `make vars`, `make pyclean`, and `make dev-install`.

.PHONY: test lint base-test script-test dist-test docker-test

# The main module of the project is usually the same as the main package of the project.
# Make sure the setup the __init__.py correctly. We can then use the module to setup
# generic import and cli tests. Override the following vars as needed.
CLI_TEST     = $(PYTHON) -m $(MAIN) -h $(NOL)
IMPORT_TEST  = $(PYTHON) -c "import $(MAIN) as m; print(\\"version:\\",m.__version__,\\"tag:\\",m.__tag__)"
DIST_TEST    = $(IMPORT_TEST); $(CLI_TEST)
BASH         = bash -it -o errexit -c

# The `test` target depends on `base-test` to trigger import tests, cli tests, and linting.
# To setup yor own linting and tests, please override the `test` and `lint` targets.
test: lint base-test
lint: vars ; $(MAKEPY) lint

# Generic tests included when running `make test`.
base-test: $(SRC_FILES) pyclean vars lint
\t$(IMPORT_TEST)
\t$(MAKEPY) test --tests $(PRJ_TESTS)
\t$(CLI_TEST)

# Test the installed package scripts (CLI tools of the package)
SCRIPT_TEST = echo "please set SCRIPT_TEST = <shell-script-code> to run script tests"
script-test:
\t$(BASH) '$(SCRIPT_TEST)' $(NOL)

# run tests using default pip and python outside of the project
dist-test:
\tcd tests && python -m pytest -x .
\tcd /tmp && $(BASH) '$(DIST_TEST)' $(NOL)

# After pushing to PyPi, you want to check if you can pull and run
# your package in a clean environment. Safest bet is to use docker!
TEST_IMG     = python:$(PY)
TEST_VOLUMES = -v $(CURDIR)/tests:/tests
docker-test:
\tdocker run -it $(TEST_VOLUMES) $(TEST_IMG) 'set -o errexit; pip install $(PKG); $(DIST_TEST)'

"""

dirs.add('make')
files['make/project.mk'] = make_project_mk = """
# Generic Python Project Make-Include File
# ========================================
# Copy these mk-files to your python project for easy testing, building, and publishing.
#
# (c) Uwe Jugel, @ubunatic, License: MIT
#
# Integration and Usage
# ---------------------
# Simply `include make/makepy.mk` to include all mk files in your `Makefile`.
# Now you can use `make vars`, `make pyclean`, and `make dev-install`.

# Targets
# -------
# vars:    print all relevant vars
# pyclean: remove pyc and other cached files
# copy:    copy all mk files to a new project

.PHONY: vars pyclean pyclean-all copy

# Printing make vars can be helpful when testing multiple Python versions.
vars:
\t# Make Variables
\t# --------------
\t# CURDIR    $(CURDIR)
\t# PKG       $(PKG)
\t# MAIN      $(MAIN)
\t#
\t# PY        $(PY)
\t# PYTHON    $(PYTHON)
\t# PIP       $(PIP)
\t# MAKEPY    $(MAKEPY)
\t# WHEELTAG  $(WHEELTAG)
\t#
\t# SETUP_DIR $(SETUP_DIR)
\t# SRC_FILES $(SRC_FILES)
\t# PRJ_TESTS $(PRJ_TESTS)
\t#
\t# Python/Pip Versions
\t# ---------------
\t# python: $(shell python    --version 2>&1)
\t# PYTHON: $(shell $(PYTHON) --version 2>&1)
\t# pip:    $(shell pip       --version 2>&1)
\t# PIP:    $(shell $(PIP)    --version 2>&1)

pyclean:
\t@pyclean . $(NEL)                          || echo 'WARNING: pyclean command failed'
\t@find . -name '*.py[co]'    -delete $(NEL) || echo 'WARNING: failed to cleanup some .py* files'
\t@find . -name '__pycache__' -delete $(NEL) || echo 'WARNING: failed to cleanup some __pycache__ dirs'

pyclean-all: pyclean
\trm -rf .pytest_cache .cache dist build backport

copy:
\ttest -n "$(TRG)"           # ensure the copy target TRG is set
\tmkdir -p $(TRG)/make       # create TRG/make dir in target dir
\tcp make/*.mk $(TRG)/make/  # copy all mk files to TRG/make

"""

dirs.add('make')
files['make/twine.mk'] = make_twine_mk = """
# Generic Python Project Make-Include File
# ========================================
# Copy these mk-files to your python project for easy testing, building, and publishing.
#
# (c) Uwe Jugel, @ubunatic, License: MIT
#
# Integration and Usage
# ---------------------
# Simply `include make/makepy.mk` to include all mk files in your `Makefile`.
# Now you can use `make vars`, `make pyclean`, and `make dev-install`.

# This file includes twine targets for publishing a package to PyPi.

.PHONY: sign test-publish tag publish

sign:
\t# sign the dist with your gpg key
\tgpg --detach-sign -a dist/*.whl

test-publish:
\t# upload to testpypi (needs valid ~/.pypirc)
\ttwine upload --repository testpypi dist/*

# TODO: add release note support
PKG_VERSION = $(shell $(MAKEPY) version)
tag: clean dists
\tgit add $(PKG)/__init__.py
\tgit commit --amend --no-edit --allow-empty
\tgit tag -f v$(PKG_VERSION)

# TODO: add support for universal wheels
PY2_WHEEL = $(shell find dist -name '$(PKG)*py2-none-any*.whl')
PY3_WHEEL = $(shell find dist -name '$(PKG)*py3-none-any*.whl')
publish: sign
\t# upload to pypi (requires pypi account)
\t# p3-wheel: $(PY3_WHEEL)
\t# p2-wheel: $(PY2_WHEEL)
\t@bash -c 'read -n 1 -p "start upload (y/N)? " key && test "$$key" = "y"'
\ttwine upload --repository pypi $(PY3_WHEEL) $(PY2_WHEEL)

"""

dirs.add('make')
files['make/makepy.mk'] = make_makepy_mk = """
# Generic Python Project Make-Include File
# ========================================
# Copy these mk-files to your python project for easy testing, building, and publishing.
#
# (c) Uwe Jugel, @ubunatic, License: MIT
#
# Integration and Usage
# ---------------------
# Simply `include make/makepy.mk` to include all mk files in your `Makefile`.
# Now you can use `make vars`, `make pyclean`, and `make dev-install`.

# This file provides targets that wrap the most common makepy commands, so that you
# can call makepy commands and custom targets together: `make <makepy-command> <custom-target>`.
# It also includes all other mk files so that you only need to `include make/makepy.mk`

.PHONY: dist backport dists install dev-install tox bumpversion version

include make/vars.mk

ifeq ($(WHEELTAG), py2)
TEST_DEP=backport
endif
test: $(TEST_DEP)

include make/project.mk
include make/tests.mk
include make/twine.mk

# passthrough makepy commands
MAKEPY_COMMANDS = dist backport dists install dev-install tox bumpversion version
$(MAKEPY_COMMANDS): ; $(MAKEPY) $@

# TODO: handle egg removal in makepy
uninstall: ; $(MAKEPY) uninstall && rm -rf *.egg-info

"""
