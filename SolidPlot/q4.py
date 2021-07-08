import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

class Solid:
    def __init__(self):
        self.points = []

    def plot(self):
        pass

    def scale(self, matrix):
        self.points = matrix.dot(self.points.T).T

    def translation(self, t):
        self.points = self.points + t

class Quadrilatero(Solid):

    def __init__(self, points):
        super(Quadrilatero, self).__init__()
        self.points = points

    def __repr__(self):
        return f'Quadrilateral: (centroid: {self.centroid})'

    @property
    def arestas(self):
        arestas = np.array([
            [self.points[0], self.points[1]],
            [self.points[0], self.points[2]],
            [self.points[2], self.points[3]],
            [self.points[1], self.points[3]],
            [self.points[1], self.points[5]],
            [self.points[5], self.points[7]],
            [self.points[7], self.points[3]],
            [self.points[5], self.points[4]],
            [self.points[4], self.points[6]],
            [self.points[6], self.points[7]],
            [self.points[0], self.points[4]],
            [self.points[2], self.points[6]],
        ])

        return arestas

    @property
    def faces(self):
        faces = np.array([
            [self.points[0], self.points[1], self.points[5], self.points[4]],
            [self.points[2], self.points[6], self.points[7], self.points[3]],
            [self.points[0], self.points[2], self.points[3], self.points[1]],
            [self.points[3], self.points[1], self.points[5], self.points[7]],
            [self.points[0], self.points[4], self.points[6], self.points[2]],
            [self.points[3], self.points[7], self.points[5], self.points[1]]
        ])

        return faces

    @property
    def centroid(self):
        return np.mean(self.points, axis = 0)

    @property
    def i(self):
        return self.points.min()

    @property
    def f(self):
        return self.points.max()

    def plot(self, center = True):
        # print(self)
        i = self.points.min()
        f = self.points.max()
        k = f
        axis_values = np.arange(i, k+1, 0.1)
        zeros = np.zeros(axis_values.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(i, f)
        ax.set_xlabel('X-label')
        ax.set_ylim3d(f, i)
        ax.set_ylabel('Z-label')
        ax.set_zlim3d(i, f)
        ax.set_zlabel('Y-label')
        ax.add_collection3d(Poly3DCollection(self.faces, facecolors='black', linewidths=1, edgecolors='r', alpha=.25))
        ax.plot3D(axis_values, zeros, zeros, c='black')
        ax.plot3D(zeros, axis_values, zeros, c='black')
        ax.plot3D(zeros, zeros, axis_values, c='black')
        # if center:
        # 	print(self.centroid)
        plt.show()


class Piramide(Solid):

    def __init__(self, points):
        super(Piramide, self).__init__()
        self.points = points

    def __repr__(self):
        return f'Pyramid: (centroid: {self.centroid})'

    @property
    def arestas(self):
        arests = np.array([
            [self.points[0], self.points[1]],
            [self.points[1], self.points[2]],
            [self.points[2], self.points[3]],
            [self.points[3], self.points[0]],
            [self.points[0], self.points[4]],
            [self.points[1], self.points[4]],
            [self.points[2], self.points[4]],
            [self.points[3], self.points[4]]
        ])

        return arests

    @property
    def faces(self):
        faces = np.array([
            [self.points[0], self.points[1], self.points[2], self.points[3]],
            [self.points[0], self.points[1], self.points[4]],
            [self.points[1], self.points[2], self.points[4]],
            [self.points[2], self.points[3], self.points[4]],
            [self.points[3], self.points[0], self.points[4]]
        ])

        return faces

    @property
    def centroid(self):
        # base_points = self.points[0:4]
        # apex = self.points[4]
        #
        # sum_x = np.sum(base_points)
        # sum_y = np.sum(base_points)
        # sum_z = np.sum(base_points)
        # base_center = np.array([sum_x, sum_y, sum_z]) / base_points.shape[0]
        #
        # center = (apex - base_center) / 4
        # return center

        return np.mean(self.points, axis = 0)

    @property
    def i(self):
        return self.points.min()

    @property
    def f(self):
        return self.points.max()

    def plot(self, center = True):
        # print(self)
        i = self.points.min()
        f = self.points.max()

        k = f
        axis_values = np.arange(i, k + 1, 0.1)
        zeros = np.zeros(axis_values.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(i, f)
        ax.set_xlabel('X-label')
        ax.set_ylim3d(f, i)
        ax.set_ylabel('Z-label')
        ax.set_zlim3d(i, f)
        ax.set_zlabel('Y-label')
        ax.add_collection3d(Poly3DCollection(self.faces, facecolors='black', linewidths=1, edgecolors='r', alpha=.25))
        ax.plot3D(axis_values, zeros, zeros, c='black')
        ax.plot3D(zeros, axis_values, zeros, c='black')
        ax.plot3D(zeros, zeros, axis_values, c='black')
        # if center:
        # 	print(self.centroid)
        plt.show()

class Mundo:

    def __init__(self):
        self.solids = []
        self.base_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def add_solid(self, solid):
        self.solids.append(solid)

    @property
    def f(self):
        p = []
        for i in range(len(self.solids)):
            p += [self.solids[i].i]
            p += [self.solids[i].f]
        p = np.array(p)
        return p.max()

    @property
    def i(self):
        p = []
        for i in range(len(self.solids)):
            p += [self.solids[i].i]
            p += [self.solids[i].f]
        p = np.array(p)
        return p.min()

    @property
    def center(self):
        at = np.array([0, 0, 0], dtype=np.float32)
        # for s in self.solids:
        # 	at += s.centroid

        at = self.solids[0].centroid + self.solids[2].centroid
        # return at / len(self.solids)
        return at / 2

    def change_base(self, eye):
        at = self.center
        n = (at - eye)
        n = np.divide(n,np.linalg.norm(n))

        # create random vector aux
        x = random.uniform(1, 5)
        y = random.uniform(1, 5)
        z = 5
        aux = np.array([x, y, -z])
        aux = np.divide(aux,np.linalg.norm(aux))

        # get perpendicular vector "v"
        v = np.cross(n,aux)
        v = np.divide(v,np.linalg.norm(v))
        # get the final vector "s"
        s = np.cross(n,v)
        s = np.divide(s,np.linalg.norm(s))


        # get new base
        self.base_vectors = np.array([
            v, n, s
        ])
        R = self.base_vectors.copy()
        # compute inverse matrix
        R = np.linalg.inv(R)
        # transform points to new coordinates
        for idx, solid in enumerate(self.solids):
            self.solids[idx].points = R.dot(self.solids[idx].points.T).T



    def plot(self, changed_base = False, eye = [0,0,0]):


        # n = n / n.dot(n)
        # exit(1)

        if not changed_base:
            self.solids[0].points = self.solids[0].points + [2,-3, 0]
            self.solids[2].points = self.solids[2].points + [6,-2, 0]

            self.solids[1].points = self.solids[1].points + [-6,5,2]
            self.solids[3].points = self.solids[3].points + [-2,5,2]

        i = min(self.solids[0].points.min(), self.solids[2].points.min())
        f = max(self.solids[0].points.max(), self.solids[2].points.max())

        n = self.center - eye
            # exit(1)

        if not changed_base:
            axis_values = np.arange(int(self.i), int(self.f) + 1, 0.1)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim3d(int(self.i-1), int(self.f+1))
            ax.set_xlabel('X-label')
            ax.set_ylim3d(int(self.f+1), int(self.i-1))
            ax.set_ylabel('Z-label')
            ax.set_zlim3d(int(self.i-1), int(self.f+1))
            ax.set_zlabel('Y-label')
        elif changed_base:
            axis_values = np.arange(int(self.i), int(self.f) + 1, 0.1)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim3d(int(i), int(f))
            ax.set_xlabel('X-label')
            ax.set_ylim3d(int(f), int(i))
            ax.set_ylabel('Z-label')
            ax.set_zlim3d(int(i), int(f))
            ax.set_zlabel('Y-label')

            # self.solids[1].translation([-15, 0, 0])
            # self.solids[3].translation([-15, 0, 0])


        # if not changed_base:
        #     print(self.center, eye, n)
        #     ax.plot3D(self.center[0] * axis_values,
        #               self.center[1] * axis_values,
        #               self.center[2] * axis_values, c='red')
        #
        #     ax.plot3D(eye[0] * axis_values,
        #               eye[1] * axis_values,
        #               eye[2] * axis_values, c='red')
        #
        #     ax.plot3D(n[0] * axis_values,
        #               n[1] * axis_values,
        #               n[2] * axis_values, c='blue')
        if not changed_base:
            ax.plot3D(self.base_vectors[0][0] * axis_values,
                      self.base_vectors[0][1] * axis_values,
                      self.base_vectors[0][2] * axis_values, c='black')
            ax.plot3D(self.base_vectors[1][0] * axis_values,
                      self.base_vectors[1][1] * axis_values,
                      self.base_vectors[1][2] * axis_values, c='black')
            ax.plot3D(self.base_vectors[2][0] * axis_values,
                      self.base_vectors[2][1] * axis_values,
                      self.base_vectors[2][2] * axis_values, c='black')
        # ax.plot3D(zeros, axis_values, zeros, c='black')
        # ax.plot3D(zeros, zeros, axis_values, c='black')
        if not changed_base:
            for i in range(len(self.solids)):
                ax.add_collection3d(
                    Poly3DCollection(self.solids[i].faces, facecolors='black',
                                     linewidths=1, edgecolors='r', alpha=.25))

        else:
            ax.add_collection3d(
                Poly3DCollection(self.solids[0].faces, facecolors='black',
                                 linewidths=1, edgecolors='r', alpha=.25))
            ax.add_collection3d(
                Poly3DCollection(self.solids[2].faces, facecolors='black',
                                 linewidths=1, edgecolors='r', alpha=.25))

        plt.show()
        if not changed_base:
            self.solids[0].translation(n)
            self.solids[2].translation(n)


    def project(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        colors = ['b', 'g']
        self.solids[0].points = np.array([self.solids[0].points[:, 0], self.solids[0].points[:, 2]]).T
        self.solids[2].points = np.array([self.solids[2].points[:, 0], self.solids[2].points[:, 2]]).T
        # solids.append(solid)

        i = int(min(self.solids[0].points.min(), self.solids[2].points.min()))
        f = int(max(self.solids[0].points.max(), self.solids[2].points.max()))
        ax.set_xlim(i - 1, f + 1)
        ax.set_ylim(i - 1, f + 1)
        k = 0
        for i, solid in enumerate(self.solids):
            if i == 0 or i == 2:
                for aresta in solid.arestas:
                    ax.plot(aresta[:, 0], aresta[:, 1], color=colors[k])
                k+=1
        plt.show()


if __name__ == '__main__':
    center = [0,0,0]
    a = 1.5
    dx = dy = dz = a/2

    cube = np.array([
        [-dx, 0, -dz],
        [-dx, 0, dz],
        [-dx, 2*dy, -dz],
        [-dx, 2*dy, dz],
        [dx, 0, -dz],
        [dx, 0, dz],
        [dx, 2*dy, -dz],
        [dx, 2*dy, dz]
    ])
    cube[:,[1, 2]] = cube[:,[2, 1]]
    cube = Quadrilatero(cube)

    x = 1.5
    y = 5.0
    z = 2.5

    paral = np.array([
        [0, 0, 0],
        [0, 0, z],
        [0, y, 0],
        [0, y, z],
        [x, 0, 0],
        [x, 0, z],
        [x, y, 0],
        [x, y, z]
    ])

    paral[:, [1, 2]] = paral[:, [2, 1]]
    paral = Quadrilatero(paral)

    pyr = np.array([
        [-1,0,1],
        [1,0,1],
        [1, 0, -1],
        [-1, 0, -1],
        [0,3,0]
    ])
    pyr[:, [1,2]] = pyr[:, [2,1]]
    pyr = Piramide(pyr)

    k1 = 3.0
    k2 = 1.3
    h = 2.5
    trunk = np.array([
        [-k1/2, 0, -k1/2],
        [-k1/2, 0, k1/2],
        [-k2/2, h, -k2/2],
        [-k2/2, h, k2/2],

        [k1/2, 0, -k1/2],
        [k1/2, 0, k1/2],
        [k2/2, h, -k2/2],
        [k2/2, h, k2/2],
    ])

    trunk[:, [1,2]] = trunk[:, [2,1]]
    trunk = Quadrilatero(trunk)

    x = np.random.uniform(3,2)
    y = np.random.uniform(2,3)
    z = np.random.uniform(1,2)

    # [8, 1, 4]
    # [3, 2, 1]
    x = 5
    y = 1
    z = 4

    mundo = Mundo()

    mundo.add_solid(cube)
    mundo.add_solid(paral)
    mundo.add_solid(pyr)
    mundo.add_solid(trunk)
    e = [x, z, y]
    mundo.plot(eye=e)

    mundo.change_base(e)
    mundo.plot(changed_base = True)
    mundo.project()
