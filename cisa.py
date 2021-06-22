import matplotlib.pyplot as plt
import numpy as np
import math

def cis(p,c):
	return np.int32(np.dot(p,c))


def scale_p(p):
	return 10*np.array(p)

class Point:

	def __init__(self, x, y,):
		self.x = x
		self.y = y

	def __repr__(self):
		return f'Point(x: {self.x}, y: {self.y})'


class Line:

	def __init__(self, point_i, point_f):

		self.point_i = point_i
		self.point_f = point_f

	@property
	def m(self):
		return (self.point_f.y - self.point_i.y) / (self.point_f.x - self.point_i.x)

	@property
	def b(self):
		return self.point_f.y - self.m*self.point_f.x
	
	
	@property
	def points(self):
		if self.m < 0:
			self.point_i, self.point_f = self.point_f, self.point_i
		return [[[np.int32(x), np.int32(self.m*x + self.b)] for x in range(self.point_i.x, self.point_f.x)]][0]

	def __repr__(self):
		return f'Line: (pi = {self.point_i}, pf = {self.point_f}, m = {self.m})\n'

	def create_fragment(self, x, y):
	    x_m = math.floor(np.abs(x))
	    y_m = math.floor(np.abs(y))

	    return x_m, y_m

	def fill(self, matrix):
		
		for p in self.points:
			x,y = self.create_fragment(p[0],p[1])
			matrix[x][y] = 255

		matrix[self.point_i.x][self.point_i.y] = 255
		matrix[self.point_f.x][self.point_f.y] = 255
		return matrix
	


class Triangle:

	def __init__(self, p):
		p = np.array(p)
		[point_1, point_2, point_3] = p
		point_1 = Point(point_1[0], point_1[1])
		point_2 = Point(point_2[0], point_2[1])
		point_3 = Point(point_3[0], point_3[1])
		self.line_1_2 = Line(point_1, point_2)
		self.line_1_3 = Line(point_1, point_3)
		self.line_2_3 = Line(point_2, point_3)
		self.matrix = np.zeros((p.max()*3 + 1, p.max()*3 + 1))

	@property
	def line_1_2_(self):
		return self.line_1_2

	@property
	def line_1_3_(self):
		return self.line_1_3

	@property
	def line_2_3_(self):
		return self.line_2_3
	

	def fill(self):
		self.matrix = self.line_1_2.fill(self.matrix)
		self.matrix = self.line_1_3.fill(self.matrix)
		self.matrix = self.line_2_3.fill(self.matrix)

		return self.matrix

class C:

	def __init__(self, tr, tr1, tr2, tr3):
		self.triangulos = [tr, tr1, tr2, tr3]

	def subplot(self):
		matrix = self.triangulos[0].fill()
		# ------------------------------------
		fig, ax = plt.subplots(1,2)
		matrix1 = self.triangulos[1].fill()
		ax[0].imshow(matrix.T, origin='lower')
		ax[1].imshow(matrix1.T, origin='lower')

		plt.show()


		# ------------------------------------
		fig, ax = plt.subplots(1,2)
		matrix2 = self.triangulos[2].fill()
		ax[0].imshow(matrix.T, origin='lower')
		ax[1].imshow(matrix2.T, origin='lower')

		plt.show()

		# ------------------------------------
		fig, ax = plt.subplots(1,2)
		matrix3 = self.triangulos[3].fill()
		ax[0].imshow(matrix.T, origin='lower')
		ax[1].imshow(matrix3.T, origin='lower')

		plt.show()



if __name__ == '__main__':

	p = [

		[1,1],
		[3,1],
		[2,3]

	]

	C1 = [

		[1, 0.2],
		[0,   1]

	]

	C2 = [

		[1,   0],
		[0.3, 1]

	]

	C3 = [

		[1, 0.2],
		[0.3, 1]

	]

	p = scale_p(p)
	tr = Triangle(p)

	p1 = cis(p,C1)
	tr1 = Triangle(p1)

	p2 = cis(p,C2)
	tr2 = Triangle(p2)

	p3 = cis(p,C3)
	tr3 = Triangle(p3)

	c = C(tr, tr1, tr2, tr3)
	c.subplot()