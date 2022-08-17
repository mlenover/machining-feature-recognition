from makepy.shell import run, rm
import logging

log = logging.getLogger(__name__)

def tox(envlist=None):
    log.info('starting tox tests for envlist: %s', envlist)
    if envlist is None: run(['tox'])
    else:               run(['tox', '-e', envlist])

def clean(): rm('.tox')
