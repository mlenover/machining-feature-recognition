import os
import subprocess

#SOLIDWORKS_LOCATION = r"C:\Program Files\SOLIDWORKS Corp\SOLIDWORKS"

def run():
    SOLIDWORKS_LOCATION = input("Please enter your Solidworks installation directory. By default, this is at C:\Program Files\SOLIDWORKS Corp\SOLIDWORKS \n>")
    wd = os.getcwd()
    makepy_dir = wd + '\modeltaggingvenv\Lib\site-packages\win32com\client'
    subprocess.run(r'modeltaggingvenv\Scripts\python "{}\makepy.py" -o swfuncs.py -v "{}\sldworks.tlb"'.format(makepy_dir, SOLIDWORKS_LOCATION))
    subprocess.run(r'modeltaggingvenv\Scripts\python "{}\makepy.py" -o swconst.py -v "{}\swconst.tlb"'.format(makepy_dir, SOLIDWORKS_LOCATION))