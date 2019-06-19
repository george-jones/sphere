from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from panda3d.core import Point3
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter, GeomTriangles, GeomNode


class MyApp(ShowBase):
 
    def __init__(self):
        ShowBase.__init__(self)
        format = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData('sphere', format, Geom.UHStatic)
        vdata.setNumRows(1)

        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')
        prim = GeomTriangles(Geom.UHStatic)

        n = 3
        vertex.addData3f(1, 5, 1)
        vertex.addData3f(-1, 5, 1)
        vertex.addData3f(-1, 5, -1)
        for i in range(0, n):
            color.addData4f(1.0, 1.0, 1.0, 1.0)

        prim.addVertices(0, 1, 2)
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode('TheSphere')
        node.addGeom(geom)
        nodePath = self.render.attachNewNode(node)

    def z__init__(self):
        ShowBase.__init__(self)

        format = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData('sphere', format, Geom.UHStatic)
        r = 10.0
        n = 8
        col_inc = pi / (2*n)
        row_inc = -1 * pi / (n)
        vdata.setNumRows(9)
        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')
        for row in range(0, n+1):
            phi = (pi / 4) + row * row_inc
            for col in range(0, n+1):
                theta = (-1 * pi / 4) + col * col_inc
                x = r * sin(phi) * cos(theta)
                y = 50 + r * cos(phi)
                z = r * sin(phi) * sin(theta)
                vertex.addData3f(x, y, z)
                color.addData4f(row/(n+1), 0, col/(n+1), 1)
        prim = GeomTriangles(Geom.UHStatic)
        for row in range(0, n):
            row_idx = row * (n+1)
            next_row_idx = (row+1) * (n+1)
            for col in range(0, n):
                col_idx = row_idx + col
                prim.addVertices(col_idx + 1, col_idx, next_row_idx + col)
                #prim.addVertices(col_idx + 1, next_row_idx + col, next_row_idx + col + 1)
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode('TheSphere')
        node.addGeom(geom)
        nodePath = self.render.attachNewNode(node)

 
app = MyApp()
app.run()