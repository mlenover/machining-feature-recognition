import win32com.client as win32

try:
    import swconst
except ImportError:
    import setup
    setup.run()
    import swconst

swconst = swconst.constants
