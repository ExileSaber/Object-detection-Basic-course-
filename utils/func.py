import os
import xml.etree.ElementTree as ET

def get_absolute_path(path):
    current_directory = os.getcwd()
    # print("当前工作目录的绝对路径:", current_directory)
    path = os.path.join(current_directory, path)
    return path



    