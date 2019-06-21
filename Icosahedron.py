from math import pi, sin, cos, sqrt
import random

from direct.showbase.ShowBase import ShowBase
from panda3d.core import Point3
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter, GeomTriangles, GeomNode

class PointGraph():
    def __init__(self):
        self.nodes = [ ]
    
    def addNode(self, x, y, z):
        pn = PointNode(x, y, z, len(self.nodes))
        self.nodes.append(pn)
    
    def connect(self, pt_idx, conn_indexes):
        p1 = self.nodes[pt_idx]
        for idx in conn_indexes:
            p1.connect(self.nodes[idx])


class PointNode():
    """A PointNode is a vertex that knows its index in the list,
    and all the other nodes it is connected to.

    """
    def __init__(self, x, y, z, idx):
        self.x = x
        self.y = y
        self.z = z
        self.idx = idx
        self.conn = [ ]

    def connect(self, pt):
        """ Make sure to connect nodes in counter-clockwise order.

        """
        if pt not in self.conn:
            self.conn.append(pt)


def pg_draw_tris(pg, render):
    format = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData('pgtris', format, Geom.UHStatic)
    vdata.setNumRows(len(pg.nodes))
    vertex = GeomVertexWriter(vdata, 'vertex')
    color = GeomVertexWriter(vdata, 'color')
    prim = GeomTriangles(Geom.UHStatic)

    for pt in pg.nodes:
        vertex.addData3f(pt.x, pt.y, pt.z)
        color.addData4f(random.random(), random.random(), random.random(), 1.0)

    for pt in pg.nodes:
        if len(pt.conn) > 0:
            for i,cpt in enumerate(pt.conn):
                next_cpt = pt.conn[(i+1) % len(pt.conn)]
                prim.addVertices(pt.idx, cpt.idx, next_cpt.idx)
                print("%d - %d - %d" % (pt.idx, cpt.idx, next_cpt.idx))

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode('TheTris')
    node.addGeom(geom)
    nodePath = render.attachNewNode(node)
    nodePath.setPos(0, 10, 0)

class MyApp(ShowBase):
 
    def __init__(self):
        ShowBase.__init__(self)

        g = (1 + sqrt(5)) / 2 # Golden ratio

        # (0, +-1, +-g) vertices at circular permutations of this
        pg = PointGraph()

        # yz plane (purple)
        pg.addNode(0, -1, g), # top front       0
        pg.addNode(0, 1, g), # top back         1
        pg.addNode(0, 1, -1*g), # bottom back   2
        pg.addNode(0, -1, -1*g), # bottom front 3
        # xy plane (light green)
        pg.addNode(-1, -1*g, 0), # front left   4
        pg.addNode(-1, g, 0), # back left       5
        pg.addNode(1, g, 0), # back right       6
        pg.addNode(1, -1*g, 0), # front right   7
        # xz plane (dark green)
        pg.addNode(-1*g, 0, 1), # top left      8
        pg.addNode(g, 0, 1), # top right        9
        pg.addNode(g, 0, -1), # bottom right    10
        pg.addNode(-1*g, 0, -1) # bottom left   11
    
        # connections to vertices on yz plane
        pg.connect(0, [1, 8, 4, 7, 9])
        pg.connect(1, [0, 9, 6, 5, 8])
        pg.connect(2, [11, 5, 6, 10, 3])
        pg.connect(3, [10, 7, 4, 11, 2])
        # connections to vertices on xy plane
        pg.connect(4, [7, 0, 8, 11, 3])
        pg.connect(5, [8, 1, 6, 2, 11])
        pg.connect(6, [5, 1, 9, 10, 2])
        pg.connect(7, [10, 9, 0, 4, 3])
        # connections to vertices on xz plane
        pg.connect(8, [4, 0, 1, 5, 11])
        pg.connect(9, [6, 1, 0, 7, 10])
        pg.connect(10, [6, 9, 7, 3, 2])
        pg.connect(11, [4, 8, 5, 2, 3])
    

        pg_draw_tris(pg, self.render)


 
app = MyApp()
app.run()