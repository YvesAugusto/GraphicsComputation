import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, LineCollection
import random

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
	def centroid(self):
		# sum_x = np.sum(self.points[:, 0])
		# sum_y = np.sum(self.points[:, 1])
		# sum_z = np.sum(self.points[:, 2])
		# center = np.array([sum_x, sum_y, sum_z]) / self.points.shape[0]
		# return center
		# ------------- Calcula arestas superior e inferior --------------
		l1 = np.abs(np.sum(self.points[0] - self.points[1]))
		l2 = np.abs(np.sum(self.points[2] - self.points[3]))
		# ------------- Calcula áreas superior e inferior --------------
		A1 = l1 ** 2
		A2 = l2 ** 2

		# ------------- Calcula os centros das bases superior e inferior --------------
		inferior_base_points = self.faces[0].copy()
		inferior_base_center = np.mean(inferior_base_points, axis = 0)
		superior_base_points = self.faces[1].copy()
		superior_base_center = np.mean(superior_base_points, axis=0)

		# ------------- Calcula o vetor de altura --------------
		h = np.abs(superior_base_center[2] - inferior_base_center[2])

		# Calcula a que altura do centro da base ficará o centroide
		factor = 0.25 * (A1 + 2 * np.sqrt(A1 * A2) + 3 * A2) / (A1 + np.sqrt(A1 * A1) + A2)
		# Adiciona o valor de altura ao eixo y do centro da base inferior
		center = inferior_base_center + np.array([0, 0, h * factor])
		return center

	@property
	def i(self):
		return self.points.min()

	@property
	def f(self):
		return self.points.max()

	def plot(self, center = True):
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
		if center:
			print(self.centroid)
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
	def centroid(self):
		base_points = self.points[0:4]
		apex = self.points[4]

		sum_x = np.sum(base_points)
		sum_y = np.sum(base_points)
		sum_z = np.sum(base_points)
		base_center = np.array([sum_x, sum_y, sum_z]) / base_points.shape[0]

		center = (apex - base_center) / 4
		return center

	@property
	def i(self):
		return self.points.min()

	@property
	def f(self):
		return self.points.max()

	def plot(self, center = True):
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
		if center:
			print(self.centroid)
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
		for s in self.solids:
			at += s.centroid

		return at / len(self.solids)

	def change_base(self, eye):
		at = self.center
		n = (at - eye) / (at - eye) ** 2
		# create random vector aux
		x = random.uniform(0.6, 0.8)
		y = random.uniform(0,0.01)
		z = random.uniform(1.6, 1.8)
		aux = [-x, y, z]
		print(aux)

		# get perpendicular vector "v"
		v = np.cross(n,aux)
		v = v / v ** 2
		# get the final vector "s"
		s = np.cross(v,n)
		s = s / s ** 2

		T = self.base_vectors.copy()
		self.base_vectors = np.array([
			s, v, n
		])
		R = self.base_vectors.copy()

		for idx, solid in enumerate(self.solids):
			for i in range(self.solids[idx].points.shape[0]):
				self.solids[idx].points[i] = R.T.dot(T).dot(self.solids[idx].points[i])

	def project(self):

		axis_values = np.arange(int(self.i), int(self.f) + 1, 0.1)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		# ax = fig.add_subplot(111, projection='3d')
		# ax.set_xlim3d(int(self.i - 1), int(self.f + 1))
		# ax.set_xlabel('X-label')
		# ax.set_ylim3d(int(self.f + 1), int(self.i - 1))
		# ax.set_ylabel('Z-label')
		# ax.set_zlim3d(int(self.i - 1), int(self.f + 1))
		# ax.set_zlabel('Y-label')

		# ax.plot3D(self.base_vectors[0][0] * axis_values,
		# 		  self.base_vectors[0][1] * axis_values,
		# 		  self.base_vectors[0][2] * axis_values, c='black')
		# ax.plot3D(self.base_vectors[1][0] * axis_values,
		# 		  self.base_vectors[1][1] * axis_values,
		# 		  self.base_vectors[1][2] * axis_values, c='black')
		# ax.plot3D(self.base_vectors[2][0] * axis_values,
		# 		  self.base_vectors[2][1] * axis_values,
		# 		  self.base_vectors[2][2] * axis_values, c='black')


		colors = ['b', 'g', 'r', 'c']
		faces = []
		for i in range(len(self.solids)):
			face = self.solids[i].points.copy()
			face = np.array([face[:,0], face[:,2]]).T
			faces.append(face)
			# ax.plot(face[:,0], face[:,1], color = colors[i])
			# print(f'Face {i + 1}: {face}')


		faces = np.array(faces)
		ax.set_xlim(self.i-1, self.f+1)
		ax.set_ylim(self.i-1, self.f+1)
		ax.add_collection(
			LineCollection(faces, linewidths=(2, 2, 2, 2), colors=colors, linestyle='solid'))
		plt.show()

	def plot(self, changed_base = False):

		if not changed_base:
			self.solids[0].points = self.solids[0].points + [2,-3, 0]
			self.solids[2].points = self.solids[2].points + [5,-2, 0]

			self.solids[1].points = self.solids[1].points + [-6,3,1]
			self.solids[3].points = self.solids[3].points + [-2,3,1]

		axis_values = np.arange(int(self.i), int(self.f) + 1, 0.1)
		# x =  +  + self.base_vectors[0][1] * axis_values
		# y = self.base_vectors[1][0] * axis_values + self.base_vectors[1][1] * axis_values + self.base_vectors[1][1] * axis_values
		# z = self.base_vectors[2][0] * axis_values + self.base_vectors[2][1] * axis_values + self.base_vectors[2][1] * axis_values
		zeros = np.zeros(axis_values.shape)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlim3d(int(self.i-1), int(self.f+1))
		ax.set_xlabel('X-label')
		ax.set_ylim3d(int(self.f+1), int(self.i-1))
		ax.set_ylabel('Z-label')
		ax.set_zlim3d(int(self.i-1), int(self.f+1))
		ax.set_zlabel('Y-label')

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

	e = [-1, -3, -1]
	mundo = Mundo()
	mundo.add_solid(cube)
	mundo.add_solid(paral)
	mundo.add_solid(pyr)
	mundo.add_solid(trunk)
	mundo.plot()

	mundo.change_base(e)
	mundo.plot(changed_base = True)

	mundo.project()