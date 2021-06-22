import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def get_faces_from_cube(cube):
	face_xz = [
		cube[0],
		cube[1],
		cube[5],
		cube[4]
	]
	face_op_xz = [
		cube[2],
		cube[6],
		cube[7],
		cube[3],
	]

	face_yz = [
		cube[0],
		cube[2],
		cube[3],
		cube[1],
	]

	face_op_yz = [
		cube[3],
		cube[1],
		cube[5],
		cube[7]
	]

	face_xy = [
		cube[0],
		cube[4],
		cube[6],
		cube[2]
	]

	face_op_xy = [
		cube[3],
		cube[7],
		cube[5],
		cube[1]
	]

	cube = np.array([
		face_xz, face_op_xz,
		face_yz, face_op_yz,
		face_xy, face_op_xy
	])

	return cube

if __name__ == '__main__':
	center = [0,0,0]
	a = 1.5
	dx = dy = dz = a/2

	# cube = np.array([
	# 	[0, 0, 0],
	# 	[0, 0, 1],
	# 	[0, 1, 0],
	# 	[0, 1, 1],
	# 	[1, 0, 0],
	# 	[1, 0, 1],
	# 	[1, 1, 0],
	# 	[1, 1, 1]
	# ])

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

	P = np.array([[1, 0.4, 0],
		 [0, 1, 0],
		 [0, 0, 1]])

	P2 = np.array([[1, 0, 0],
				  [0.4, 1, 0],
				  [0, 0, 1]])
	# -------------------------------------------

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	r = [-1, 1]
	X, Y = np.meshgrid(r, r)
	one = np.array([[1]])
	minus = np.array([[-1]])
	ax.plot_surface(X, Y, one, alpha=0.01)
	ax.plot_surface(X, Y, minus, alpha=0.01)
	ax.plot_surface(X, minus, Y, alpha=0.01)
	ax.plot_surface(X, one, Y, alpha=0.01)
	ax.plot_surface(one, X, Y, alpha=0.01)
	ax.plot_surface(minus, X, Y, alpha=0.01)
	normal_cube = get_faces_from_cube(cube)
	ax.add_collection3d(Poly3DCollection(normal_cube, facecolors='k', linewidths=1, edgecolors='r', alpha=.25))
	plt.show()

	# -------------------------------------------
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	r = [-1, 1]
	X, Y = np.meshgrid(r, r)
	one = np.array([[1]])
	minus = np.array([[-1]])
	ax.plot_surface(X, Y, one, alpha=0.01)
	ax.plot_surface(X, Y, minus, alpha=0.01)
	ax.plot_surface(X, minus, Y, alpha=0.01)
	ax.plot_surface(X, one, Y, alpha=0.01)
	ax.plot_surface(one, X, Y, alpha=0.01)
	ax.plot_surface(minus, X, Y, alpha=0.01)
	cis1_cube = get_faces_from_cube(cube.dot(P))
	ax.add_collection3d(Poly3DCollection(normal_cube, facecolors='w', linewidths=1, edgecolors='black', alpha=.5))
	ax.add_collection3d(Poly3DCollection(cis1_cube, facecolors='black', linewidths=1, edgecolors='black', alpha=.25))
	plt.show()

	# -------------------------------------------
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	r = [-1, 1]
	X, Y = np.meshgrid(r, r)
	one = np.array([[1]])
	minus = np.array([[-1]])
	ax.plot_surface(X, Y, one, alpha=0.01)
	ax.plot_surface(X, Y, minus, alpha=0.01)
	ax.plot_surface(X, minus, Y, alpha=0.01)
	ax.plot_surface(X, one, Y, alpha=0.01)
	ax.plot_surface(one, X, Y, alpha=0.01)
	ax.plot_surface(minus, X, Y, alpha=0.01)
	cis2_cube = get_faces_from_cube(cube.dot(P2))
	ax.add_collection3d(Poly3DCollection(normal_cube, facecolors='w', linewidths=1, edgecolors='black', alpha=.5))
	ax.add_collection3d(Poly3DCollection(cis2_cube, facecolors='black', linewidths=1, edgecolors='black', alpha=.25))
	plt.show()
