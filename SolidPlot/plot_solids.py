import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

class Solid:
	def __init__(self):
		pass

	def plot(self):
		pass

class Quadrilatero(Solid):

	def __init__(self, points):
		super(Quadrilatero, self).__init__()
		self.points = points

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
	def i(self):
		return self.points.min()

	@property
	def f(self):
		return self.points.max()

	def plot(self):
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
		plt.show()


class Piramide(Solid):

	def __init__(self, points):
		super(Piramide, self).__init__()
		self.points = points

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
	def i(self):
		return self.points.min()

	@property
	def f(self):
		return self.points.max()

	def plot(self):
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
		plt.show()

class Mundo:

	def __init__(self):
		self.solids = []
		self.origo = [0,0,0]

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

	def plot(self):


		self.solids[0].points = self.solids[0].points + [0,-5,-2]
		self.solids[2].points = self.solids[2].points + [3,-5,-2]

		self.solids[1].points = self.solids[1].points + [-5,3,1]
		self.solids[3].points = self.solids[3].points + [-1, 3, 1]
		axis_values = np.arange(self.i, self.f + 1, 0.1)
		zeros = np.zeros(axis_values.shape)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlim3d(self.i, self.f)
		ax.set_xlabel('X-label')
		ax.set_ylim3d(self.f, self.i)
		ax.set_ylabel('Z-label')
		ax.set_zlim3d(self.i, self.f)
		ax.set_zlabel('Y-label')

		ax.plot3D(axis_values, zeros, zeros, c='black')
		ax.plot3D(zeros, axis_values, zeros, c='black')
		ax.plot3D(zeros, zeros, axis_values, c='black')
		for i in range(len(self.solids)):
			ax.add_collection3d(
				Poly3DCollection(self.solids[i].faces, facecolors='black',
								 linewidths=1, edgecolors='r', alpha=.25))
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
	cube.plot()

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
	paral.plot()

	pyr = np.array([
		[-1,0,1],
		[1,0,1],
		[1, 0, -1],
		[-1, 0, -1],
		[0,3,0]
	])
	pyr[:, [1,2]] = pyr[:, [2,1]]
	pyr = Piramide(pyr)
	pyr.plot()

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
	trunk.plot()

	mundo = Mundo()
	mundo.add_solid(cube)
	mundo.add_solid(paral)
	mundo.add_solid(pyr)
	mundo.add_solid(trunk)
	mundo.plot()
