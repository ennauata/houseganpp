import random
import os
from shutil import copy
import glob
from xml.dom import minidom


def edit_svg(svg_path, dst_path):
    xmldoc = minidom.parse(svg_path)
    polys = xmldoc.getElementsByTagName("polygon")
    for p in polys:
        if p.getAttribute('fill') not in ["#D3A2C7", "#727171"]:
            p.setAttribute("stroke-width", "4")
        else:
            p.setAttribute("stroke-width", "0")
    svg_str = xmldoc.toxml()
    with open(dst_path, 'w') as f:
        f.write(svg_str)
    return


files_svg = glob.glob('./*')
for _file in files_svg:
    edit_svg(_file, _file.replace('.svg', '_new.svg'))