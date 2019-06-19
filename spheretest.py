from math import sin, cos, pi, sqrt, pow

def spherical_to_cart(theta, phi, r):
    return (r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi))


def dist(x1, y1, z1, x2, y2, z2):
    return sqrt(pow(x2-x1, 2.0) + pow(y2-y1, 2.0) + pow(z2-z1, 2.0))


def main():
    a1 = (-1 * pi / 4, pi / 4)
    a2 = (-3 * pi / 16, 3 * pi / 16)
    #b1 = (-1 * pi / 32, pi / 32)
    #b2 = (pi / 32, -1 * pi / 32)
    b1 = (-1 * pi / 4, 3 * pi / 16)
    b2 = (-3 * pi / 16, 2 * pi / 16)
    x1, y1, z1 = spherical_to_cart(a1[0], a1[1], 1.0)
    x2, y2, z2 = spherical_to_cart(a2[0], a2[1], 1.0)
    x3, y3, z3 = spherical_to_cart(b1[0], b1[1], 1.0)
    x4, y4, z4 = spherical_to_cart(b2[0], b2[1], 1.0)
    d1 = dist(x1, y1, z1, x2, y2, z2)
    d2 = dist(x3, y3, z3, x4, y4, z4)
    print(d1)
    print(d2)




main()