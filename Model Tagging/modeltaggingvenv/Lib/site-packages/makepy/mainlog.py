import logging, sys  # noqa:F401

try:
    import structlog
    from structlog.stdlib import ProcessorFormatter  # noqa: F401
    # ProcessorFormatter is available in newer versions of structlog, using this import
    # we can force fail older versions of structlog.
except ImportError: structlog = None
try: import colorama
except ImportError: colorama = None
try: import pythonjsonlogger
except ImportError: pythonjsonlogger = None

log = logging.getLogger(__name__)

PY2 = sys.version_info.major < 3

CONSOLE = 'console'
JSON    = 'json'

def configure_jsonlog():
    if PY2: coder = structlog.processors.UnicodeEncoder()
    else:   coder = structlog.processors.UnicodeDecoder()
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            coder,
            structlog.stdlib.render_to_log_kwargs,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def configure_consolelog():
    structlog.configure(
        processors = [
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory = structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use = True,
    )

def setup_structlog(level=logging.INFO, mode=CONSOLE):
    """setup_structlog sets up colored structlog logging and
    adds a structlog formatter to stdlib logging.
    """

    if structlog is None:
        logging.basicConfig(level=level)
        log.error(SystemError("please install structlog"))
        return

    use_colors = colorama is not None

    if   mode == JSON:    configure_jsonlog()
    elif mode == CONSOLE: configure_consolelog()
    else: raise ValueError("invalid mode: {}".format(mode))

    if mode == JSON:
        from pythonjsonlogger import jsonlogger
        stream = sys.stdout
        formatter = jsonlogger.JsonFormatter()
    else:
        stream = sys.stderr
        formatter = structlog.stdlib.ProcessorFormatter(
            processor = structlog.dev.ConsoleRenderer(colors=use_colors),
            foreign_pre_chain = [
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
            ]
        )

    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)

    _log = structlog.get_logger(__name__)
    _log.debug("setup structlog level={}".format(logging.getLevelName(level)),
               use_colors=use_colors)

def shutdown():
    root = logging.getLogger()
    for h in root.handlers[:]: root.removeHandler(h)
    for f in root.filters[:]:  root.removeFilter(f)

def setup_logging(level=logging.INFO, use_structlog=False, mode=CONSOLE):
    """setup_logging sets up stdlib or structlog logging with to the given level.
    If use_structlog is True, it also sets up structlog logging with convenient defaults
    for human-readable logs.
    """
    shutdown()

    if mode == JSON:
        assert structlog         is not None  # structlog required for json logging
        assert pythonjsonlogger  is not None  # pythonjsonlogger required for json logging
        use_structlog = True
    if use_structlog:
        setup_structlog(level=level, mode=mode)
    else:
        logging.basicConfig(level=level)
        logging.getLogger().setLevel(level)

    log.debug('setup logging, level=%s', logging.getLevelName(level))
