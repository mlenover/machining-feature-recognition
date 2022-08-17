from __future__ import absolute_import

import json, logging
import makepy as _makepy
import makepy.commands as _cmd
from makepy import argparse
from makepy.tox import tox, clean
from makepy.config import read_setup_args, read_basic_cfg, module_name, package_dir, package_name
from makepy.shell import pyv, wheeltag
from os.path import abspath, basename

log = logging.getLogger('makepy')

def main(argv=None):
    # 1. setup defaults for often required options
    template_src = ['setup.cfg', 'makepy.cfg', 'tox.ini', 'README.md', 'LICENSE.txt']
    common_src = list(_cmd.datadirs) + list(_cmd.datafiles) + template_src
    src = [basename(abspath('.')).split('.')[0]] + common_src
    # 2. create the parser with common options
    p = argparse.MakepyParser().with_logging(use_structlog=True).with_debug().with_protected_spaces()
    # 3. setup all flags, commands, etc. as aligned one-liners
    p.opti('commands',    help='makepy command', nargs='*', metavar='CMD')
    p.flag('tox',         help='run tox tests')
    p.flag('backport',    help='backport project to Python 2')
    p.flag('uninstall',   help='uninstall package from all pips')
    p.flag('clean',       help='clean build files')
    p.flag('test',        help='run unit tests directly')
    p.flag('lint',        help='lint source code')
    p.flag('dist',        help='build the wheel')
    p.flag('dists',       help='build all the wheels')
    p.flag('install',     help='build and install the wheel')
    p.flag('dev-install', help='directly install the source code in the current environment')
    p.flag('init',        help='create makepy files in a new project')
    p.flag('include',     help='print makepy Makefile location (usage: `include $(shell makepy include)`)')
    p.flag('path',        help='print PYHTONPATH to use makepy as module')
    p.flag('format',      help='format python source files using makepy column-aligned formatter')
    p.flag('embed',       help='embed plain text files into pythom code')
    p.flag('version',     help='show version of developed package')
    p.flag('bumpversion', help='increase patch version of package')
    p.flag('setupargs',   help='run read_setup_args in current dir and return as JSON')

    p.opti('--py',      '-P', help='set python version  CMD: test, lint, install, uninstall', default=pyv(), type=int)
    p.opti('--pkg',     '-p', help='set package name    CMD: init, install, uninstall')
    p.opti('--src',     '-s', help='set source files    CMD: backport', nargs='*', default=src)
    p.opti('--trg',     '-t', help='set target dir      CMD: init, copy-tools, dist-install', default='.')
    p.opti('--input',   '-i', help='input files         CMD: embed', nargs='*', default=[])
    p.opti('--output',  '-o', help='output file         CMD: embed', default='-')
    p.opti('--tests',   '-T', help='set tests dir       CMD: test')
    p.opti('--main',    '-m', help='set main module     CMD: init, backport')
    p.opti('--envlist', '-e', help='set tox envlist     CMD: init, tox')
    p.flag('--force',   '-f', help='overwrite files     CMD: init, copy-tools')
    p.flag('--mkfiles', '-k', help='create mk-files     CMD: init, copy_tools')
    p.flag('--version',       help='show makepy installaton info', dest='version_info')
    # additional help should be available via the 'help' command that
    # provides help for all or some ot the next given option
    # p.flag('help',            help='show contextual help')

    # 4. parse, analyze, and repair args
    args = p.parse_args(argv)

    # repair missing args from related args
    trg  = abspath(args.trg)  # create a valid named path
    pkg  = args.pkg
    main = args.main

    if pkg  is None: pkg = basename(trg)     # derive package name from target path if not given
    module = module_name(pkg)                # derive moduile name from package argument
    if main is None: main = module + '.cli'  # set default main module
    pkg_name = package_name(pkg)             # the actual package name may differ from the package argument

    # trying to use values from setup.cfg
    cfg = read_basic_cfg(trg)
    if 'name' in cfg:
        pkg_name = cfg['name']                               # name must be in cfg
        module   = cfg.get('module', module_name(pkg_name))  # get or derive module from name
        main     = cfg.get('main',   module + '.cli')        # get or set default main module

    pkg_dir  = package_dir(module)

    log.debug('using makepy config: %s:', {
        'args.trg': args.trg,
        'args.pkg': args.pkg,
        'args.py':  args.py,
        'pkg_name': pkg_name,
        'pkg_dir':  pkg_dir,
        'module':   module,
        'main':     main,
        'from_setup_cfg': 'name' in cfg,
    })

    if args.version_info:
        print(_makepy.__name__, _makepy.__version__, _makepy.__tag__)
        return

    # decide what to do when run without any command
    commands = args.commands
    if 'help' in commands: help(commands); return
    if len(commands) == 0: tox(wheeltag());  return

    # add depending commands
    commands = _cmd.add_requirements(commands, py=args.py)

    # 5. run all passed commands with their shared flags and args
    for cmd in commands:
        if   cmd == 'backport':    _cmd.backport(args.src, module)
        elif cmd == 'tox':         tox(args.envlist)
        elif cmd == 'uninstall':   _cmd.uninstall(pkg_name, py=args.py)
        elif cmd == 'clean':       clean()
        elif cmd == 'setupargs':   print(json.dumps(read_setup_args()))
        elif cmd == 'include':     print(_cmd.include())
        elif cmd == 'path':        print(_cmd.makepypath())
        elif cmd == 'copy':        _cmd.copy_tools(trg, force=args.force, mkfiles=args.mkfiles)
        elif cmd == 'test':        _cmd.test(tests=args.tests, py=args.py)
        elif cmd == 'dist':        _cmd.dist(py=args.py)
        elif cmd == 'dists':       _cmd.dists(py=args.py)
        elif cmd == 'install':     _cmd.install(pkg_name, py=args.py, source=False)
        elif cmd == 'dev-install': _cmd.install(pkg_name, py=args.py, source=True)
        elif cmd == 'lint':        _cmd.lint(py=args.py)
        elif cmd == 'init':        _cmd.init(trg, pkg_name, module, main,
                                             envlist=args.envlist, force=args.force, mkfiles=args.mkfiles)
        elif cmd == 'format':      format(args.src, force=args.force)
        elif cmd == 'embed':       _cmd.embed(args.input, args.output, force=args.force)
        elif cmd == 'bumpversion': _cmd.bumpversion(pkg_dir)
        elif cmd == 'version':     _cmd.version(pkg_dir)
        else:                      raise ValueError('invalid command: {}'.format(cmd))

if __name__ == '__main__': main()
