import os

MAKEPY_LOCATION = r"C:\Users\MLeno\AppData\Local\Programs\Python\Python310\Lib\site-packages\win32com\client"
SOLIDWORKS_LOCATION = r"C:\Program Files\SOLIDWORKS Corp\SOLIDWORKS"

def run():
    os.system(r'python {}\makepy.py -o swconst.py -v "{}\swconst.tlb"'.format(MAKEPY_LOCATION, SOLIDWORKS_LOCATION))
    os.system(r'python {}\makepy.py -o swfuncs.py -v "{}\sldworks.tlb"'.format(MAKEPY_LOCATION, SOLIDWORKS_LOCATION))