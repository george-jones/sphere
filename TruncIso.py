from math import pi, sin, cos, sqrt, pow
import random

from direct.showbase.ShowBase import ShowBase
from panda3d.core import Point3
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter, GeomTriangles, GeomNode


class Shape():
    def __init__(self):
        self.faces = [ ]
        self.edges = [ ]
        self.vertices = [ ]
 
    def get_bridge_edges(self):
        """Get edges that are not part of any faces - those
        that were made to bridge two faces.

        """
        return [ e for e in self.edges if len(e.faces) == 0]

    def load_from_point_graph(self, pg):
        """Loads up vertices and edges only.  The original faces don't matter.

        """
        # Convert point nodes to vertices
        for n in pg.nodes:
            v = Vertex(n.x, n.y, n.z)
            self.vertices.append(v)
        # Make edges
        for i, n in enumerate(pg.nodes):
            v1 = self.vertices[i]
            for c in n.conn:
                v2 = self.vertices[c.idx]
                found = False
                for e_existing in self.edges:
                    if (e_existing.v1 == v1 or e_existing.v2 == v1) and \
                        (e_existing.v1 == v2 or e_existing.v2 == v2):
                        found = True
                        break
                if not found:
                    e = Edge(v1, v2)
                    self.edges.append(e)
                    # vertices know which edges they are in
                    v1.edges.append(e)
                    v2.edges.append(e)

    def spherize_vertices(self):
        """Move all points to the unit sphere

        """
        origin = Point3(0, 0, 0)
        for vert in self.vertices:
            pt = vert.pt
            vec = pt - origin
            vec.normalize()
            vert.pt = Point3(vec.x, vec.y, vec.z)


class Face():
    def __init__(self):
        self.edges = [ ]
        self.original_vertex = None

    def get_vertices(self):
        """Get unique list of vertices included in edges.

        """
        vertices = [ ]
        for edge in self.edges:
            if edge.v1 not in vertices:
                vertices.append(edge.v1)
            if edge.v2 not in vertices:
                vertices.append(edge.v2)
        return vertices

    def get_midpoint(self):
        center = Point3(0, 0, 0)
        verts = self.get_vertices()
        for v in verts:
            center += v.pt
        center /= len(verts)
        return center

    def get_points(self):
        verts = self.get_vertices()
        points = [ ]
        for v in verts:
            pt = v.pt
            if pt not in points:
                points.append(pt)
        if self.original_vertex is not None:
            center = self.original_vertex.pt
        else:
            center = Point3(0, 0, 0)
            for p in points:
                center += p
            center /= len(points)
        return { "points": sort_pts_angular(points, center), "center": center }


class Edge():
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.faces = [ ]


class Vertex():
    def __init__(self, x, y, z):
        self.pt = Point3(x, y, z)
        self.edges = [ ]
        self.spawned_face = None

    def get_connections(self, exclude_edge=None):
        """Returns array of (vertex, edge) tuples

        """
        connections = [ ]
        for e in self.edges:
            if e is not exclude_edge:
                if e.v1 != self:
                    connections.append((e.v1, e))
                if e.v2 != self:
                    connections.append((e.v2, e))
        return connections


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
        self.spawned_face = None

    def connect(self, pt):
        """ Make sure to connect nodes in counter-clockwise order.

        """
        if pt not in self.conn:
            self.conn.append(pt)


def distsq(p1, p2):
    """Get squared distance between two points.
    A bit more efficient than taking the square root when we're just interested
    in distance for comparison sake.
    
    """
    return pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2)


def find_first_ccw(p1, p2, c):
    o = Point3(0, 0, 0)
    # Vector from origin to face center
    cvec = c - o
    # Vector from face center to point 1
    p1c_vec = p1 - c
    # Vector from face center to point 2
    p2c_vec = p2 - c
    # Cross product of two vectors
    cross = p1c_vec.cross(p2c_vec)
    # Dot of the cross and origin-face vector.
    # Positive if they are in the same direction.
    dp = vec_dot(cross, cvec)
    if dp >= 0:
        # The edge is defined in a counter-clockwise fashion
        return p1
    else:
        return p2


def unsigned_angle(p1, p2, center):
    o = Point3(0, 0, 0)
    # Vector from origin to face center
    cvec = center - o
    cvec.normalize()
    # Vector from face center to point 1
    p1c_vec = p1 - center
    p1c_vec.normalize()
    # Vector from face center to point 2
    p2c_vec = p2 - center
    p2c_vec.normalize()
    a = p1c_vec.signedAngleRad(p2c_vec, cvec)
    if a >= 0:
        return a
    else:
        return 2 * pi + a


def sort_verts_angular(new_verts, center):
    # arbitrarily choose 1st vertex as one for comparison
    pt = new_verts[0].pt
    new_verts.sort(key=lambda vert: unsigned_angle(pt, vert.pt, center))


def sort_pts_angular(points, center):
    # arbitrarily choose 1st vertex as one for comparison
    pt = points[0]
    return sorted(points, key=lambda p: unsigned_angle(pt, p, center))


def find_closest_vert(verts, other_vert):
    other_pt = other_vert.pt
    a = sorted(verts, key=lambda v: distsq(v.pt, other_pt))
    return a[0]


def shape_make_face_bridges(s):
    for face in s.faces:
        # find edges that included the original vertex of this face
        vert = face.original_vertex
        for edge in vert.edges:
            if edge.v1 == vert:
                other_face = edge.v2.spawned_face
            else:
                other_face = edge.v1.spawned_face
            # find two closest vertices in the two faces
            v_this = find_closest_vert(face.get_vertices(), other_face.original_vertex)
            v_other = find_closest_vert(other_face.get_vertices(), face.original_vertex)
            if len(v_this.edges) >= 3 or len(v_other.edges) >= 3:
                # we've already connected going the other direction
                continue
            # make bridging edge
            e = Edge(v_this, v_other)
            v_this.edges.append(e)
            v_other.edges.append(e)
            s.edges.append(e)
            if v_this not in s.vertices:
                s.vertices.append(v_this)
            if v_other not in s.vertices:
                s.vertices.append(v_other)
    # remove edges from original central verts
    for face in s.faces:
        vert = face.original_vertex
        for edge in vert.edges:
            if edge in s.edges:
                s.edges.remove(edge)
        vert.edges = [ ]


def dump_points(pts):
    for pt in pts:
        print("%f,%f,%f" % (pt.x, pt.y, pt.z))


def faces_from_bridging_edges(s):
    origin = Point3(0, 0, 0)

    def find_next_ccw(f, e, prev_vert, rej_edges):
        #print("---find_next_ccw---")
        if prev_vert is None:
            options = [ (e.v1, e.v2), (e.v2, e.v1) ]
        else:
            if e.v1 is prev_vert:
                old_vert = e.v2
            else:
                old_vert = e.v1
            options = [ (old_vert, prev_vert)]
        for vert1, vert2 in options:
            vec1 = vert2.pt - vert1.pt
            ovec = vert2.pt - origin
            #print("vert1: %f,%f,%f" % (vert1.pt.x, vert1.pt.y, vert1.pt.z))
            #print("vert2: %f,%f,%f" % (vert2.pt.x, vert2.pt.y, vert2.pt.z))
            #print("conn: %d" % len(vert2.get_connections(e)))
            for cvert, cedge in vert2.get_connections(e):
                if cedge in rej_edges:
                    # don't reconsider an edge we already decided we don't want
                    continue
                #print("cvert: %f,%f,%f" % (cvert.pt.x, cvert.pt.y, cvert.pt.z))
                if len(cedge.faces) == 2 or cvert is vert1 or cvert is vert2:
                    rej_edges.append(cedge)
                    continue
                if cedge in f.edges:
                    #print('Edge already part of face')
                    continue
                vec2 = cvert.pt - vert2.pt
                cpvec = vec1.cross(vec2)
                d = vec_dot(cpvec, ovec)
                if d > 0:
                    # counterclockwise, yay
                    #print("dot > 0, counterclockwise")
                    return cedge, cvert
                else:
                    rej_edges.append(cedge)
                    #print("nope, clockwise")
        return (None, None)

    # make hexagons
    new_hex = 0
    for edge in s.get_bridge_edges():
        if len(edge.faces) == 2:
            continue
        f = Face()
        e = edge
        rejected_edges = [ ]
        new_vert = None
        while e is not None and e not in f.edges:
            f.edges.append(e)
            if len(f.edges) == 6:
                pobj = f.get_points()
                #dump_points(pobj['points'])
                x = pobj['center'].x
                y = pobj['center'].y
                z = pobj['center'].z
                for e in f.edges:
                    if f not in e.faces:
                        e.faces.append(f)
                f.original_vertex = Vertex(x, y, z)
                f.original_vertex.spawned_face = f
                s.vertices.append(f.original_vertex)
                s.faces.append(f)
                new_hex += 1
                break
            e, new_vert = find_next_ccw(f, e, new_vert, rejected_edges)


def pg_trunciso(pg):
    s = Shape()
    s.load_from_point_graph(pg)

    all_new_verts = [ ]
    # every vertex becomes a pentagon
    for v in s.vertices:
        f = Face()
        f.original_vertex = v
        v.spawned_face = f
        new_verts = [ ]
        for e in v.edges:
            # make new vertices
            if e.v1 != v:
                v2 = e.v1
            else:
                v2 = e.v2
            vec = v2.pt - v.pt
            vec /= 3.0
            new_pt = vec + v.pt
            new_verts.append(Vertex(new_pt.x, new_pt.y, new_pt.z))
            sort_verts_angular(new_verts, v.pt)
        for i in range(0, len(new_verts)):
            v1 = new_verts[i]
            v2 = new_verts[(i+1) % len(new_verts)]
            e = Edge(v1, v2)
            v1.edges.append(e)
            v2.edges.append(e)
            e.faces.append(f)
            f.edges.append(e)
        s.faces.append(f)
        all_new_verts.extend(new_verts)
    s.vertices.extend(all_new_verts)
    
    shape_make_face_bridges(s)
    faces_from_bridging_edges(s)
    s.spherize_vertices()
    return s


def vec_dot(v1, v2):
    """Does panda3d really not have this?

    """
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def shape_tessalate(s):
    # for every face, make new vertices that are scaled in
    # relative to the midpoint.  Create new faceless edges
    # connecting to the original vertices.  Remove original
    # face from the shape's face list.
    to_remove = [ ]
    to_add = [ ]
    for f in s.faces:
        midpt = f.get_midpoint()
        mid_vertex = Vertex(midpt.x, midpt.y, midpt.z)
        new_verts = [ ]
        verts = f.get_vertices()
        for old_vert in verts:
            pt = old_vert.pt
            v = Vertex((midpt.x + pt.x) / 2, (midpt.y + pt.y) / 2, (midpt.z + pt.z) / 2)
            new_verts.append(v)
            # faceless edge connecting the original vertex to new vertex
            e = Edge(v, old_vert)
            v.edges.append(e)
            old_vert.edges.append(e)
            s.edges.append(e)
            s.vertices.append(v)
        sort_verts_angular(new_verts, midpt)
        new_face = Face()
        new_face.original_vertex = mid_vertex
        s.vertices.append(mid_vertex)
        for idx, vert in enumerate(new_verts):
            next_vert = new_verts[(idx+1) % len(new_verts)]
            e = Edge(vert, next_vert)
            vert.edges.append(e)
            next_vert.edges.append(e)
            new_face.edges.append(e)
            s.edges.append(e)
        to_remove.append(f)
        to_add.append(new_face)
        if f.original_vertex in s.vertices:
            s.vertices.remove(f.original_vertex)
    for f in to_add:
        s.faces.append(f)

    # Then destroy all original faces and edges, making their linked
    # points also no longer reference them.
    for f in to_remove:
        s.faces.remove(f)
        for e in f.edges:
            if e in s.edges:
                s.edges.remove(e)
            if e in e.v1.edges:
                e.v1.edges.remove(e)
            if e in e.v2.edges:
                e.v2.edges.remove(e)
    
    s.spherize_vertices()

    # Follow hex making procedure from the truncation step.
    #shape_make_face_bridges(s)
    faces_from_bridging_edges(s)
    s.spherize_vertices()


def shape_draw_tris(s, render):
    format = GeomVertexFormat.getV3c4()
    for face in s.faces:
        r, g, b = (random.random(), random.random(), random.random())
        p = face.original_vertex.pt
        n = len(face.edges) + 1
        vdata = GeomVertexData('facetris', format, Geom.UHStatic)
        vdata.setNumRows(n+1)
        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')
        prim = GeomTriangles(Geom.UHStatic)

        vertex.addData3f(p.x, p.y, p.z)
        color.addData4f((1+r)/2, (1+g)/2, (1+b)/2, 1)
        pobj = face.get_points()
        points = pobj['points']
        for i, pt in enumerate(points):
            vertex.addData3f(pt.x, pt.y, pt.z)
            color.addData4f(r, g, b, 1.0)
        for i in range(0, len(points)):
            idx = i+1
            idx2 = idx % len(points) + 1
            prim.addVertices(0, idx, idx2)

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode('TheTris')
        node.addGeom(geom)
        nodePath = render.attachNewNode(node)
        nodePath.setPos(0, 10, 0)

    # Edges not associated with faces.  A completed shape should
    # not have any of these.  But it's useful to see them in wireframe
    # mode during transitions.
    for edge in [e for e in s.edges if len(e.faces) == 0]:
        vdata = GeomVertexData('edgetris', format, Geom.UHStatic)
        vdata.setNumRows(3)
        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')
        prim = GeomTriangles(Geom.UHStatic)

        vertex.addData3f(edge.v1.pt.x, edge.v1.pt.y, edge.v1.pt.z)
        color.addData4f(0, 0, 0, 1)
        vertex.addData3f(edge.v1.pt.x, edge.v1.pt.y, edge.v1.pt.z)
        color.addData4f(0, 0, 0, 1)
        vertex.addData3f(edge.v2.pt.x, edge.v2.pt.y, edge.v2.pt.z)
        color.addData4f(0, 0, 0, 1)

        prim.addVertices(0, 1, 2)

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode('TheEdge')
        node.addGeom(geom)
        nodePath = render.attachNewNode(node)
        nodePath.setPos(0, 10, 0)

    # show rectangle for scale
    show_rect = False
    if show_rect:
        vdata = GeomVertexData('facetris', format, Geom.UHStatic)
        vdata.setNumRows(4)
        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')
        prim = GeomTriangles(Geom.UHStatic)

        vertex.addData3f(-1, 0, 1)
        vertex.addData3f(-1, 0, -1)
        vertex.addData3f(1, 0, -1)
        vertex.addData3f(1, 0, 1)
        color.addData4f(1,1,1,1)
        color.addData4f(1,1,1,1)
        color.addData4f(1,1,1,1)
        color.addData4f(1,1,1,1)

        prim.addVertices(0, 1, 3)
        prim.addVertices(3, 1, 2)

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode('TheRect')
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
    
        #self.wireframeOn()
        s = pg_trunciso(pg)
        shape_tessalate(s)
        shape_draw_tris(s, self.render)

 
app = MyApp()
app.run()