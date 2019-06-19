from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from panda3d.core import Point3
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter, GeomTriangles, GeomNode


def spherical_to_cart(theta, phi, r):
    return (r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi))


class MyApp(ShowBase):
 
    def __init__(self):
        ShowBase.__init__(self)

        n = 10
        ystart = 5
        r = 1

        format = GeomVertexFormat.getV3c4()        
        vdata = GeomVertexData('sphere', format, Geom.UHStatic)
        vdata.setNumRows(2*n*n)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')
        prim = GeomTriangles(Geom.UHStatic)

        for row in range(0, n):
            phi = pi/4 + (row/(n-1))*pi/2
            for col in range(0, n):
                #theta = -pi/4 + (col/(n-1))*pi/2
                theta = 5*pi/4 + (col/(n-1))*pi/2
                x,y,z = spherical_to_cart(theta, phi, r)
                y = ystart + y
                vertex.addData3f(x, y, z)
                color.addData4f(row/n, row/n, col/n, 1.0)
        for row in range(0, n-1):
            for col in range(0, n-1):
                prim.addVertices(row*n + col + 1, row*n + col, (row+1)*n + col)
                prim.addVertices(row*n + col + 1, (row+1)*n + col, (row+1)*n + col + 1)

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode('TheSphere')
        node.addGeom(geom)
        nodePath = self.render.attachNewNode(node)

 
app = MyApp()
app.run()