from spade import pyxf
#myXsb=pyxf.xsb("/usr/bin/swipl")
myXsb=pyxf.xsb("/home/prasidh/Downloads/XSB/bin/xsb")


myXsb.load("/home/prasidh/Downloads/search-source1/search/bfs2.P")

result1=myXsb.query("solve(c35_1,Solution)")


print result1
