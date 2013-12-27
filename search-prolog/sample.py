from spade import pyxf

myXsb=pyxf.xsb("/home/prasidh/Downloads/XSB/bin/xsb")

myXsb.load("/home/prasidh/Downloads/search-source1/search/dfs.P")
myXsb.load("/home/prasidh/Downloads/search-source1/search/maze.P")


result1=myXsb.query("dfs_start(c35_1,goal,Path).")


print result1
