# - Transposta -> Carolina
# - Inicializador de matriz com valores diferentes e iguais -> Victor
# - random variables (usar uma parte da hora como seed para inicializar a variável) -> Carolina
# mse -> Victor
# - Alpha fixo no início, depois implementa uma otimizador

from calendar import c
import random_gen as random

class Matrix:
    def __init__(self, array):
        self.array = array
        self.rows = len(self.array)
        self.cols = len(self.array[0])
        
        self.size = self.rows * self.cols
        
    def __str__(self):
        m = len(self.array)  # Get the first dimension
        mtxStr = ''
        for i in range(m):
            mtxStr += ('[' +
                       ' '.join(map(lambda x: '{0:8.6f}'.format(x), self.array[i])) + '] \n')
        return mtxStr

    def __add__(self, other):
        # Create a new matrix
        # Check if the other object is of type Matrix
        if isinstance(other, Matrix):
            # Add the corresponding element of 1 matrices to another
            if other.size == 1:
                other = Matrix.fill(other.array[0][0], (self.rows, self.cols))
            return Matrix([[A_row + B_row for A_row, B_row in zip(*rows)] for rows in zip(self.array, other.array)])

                # If the other object is a scaler
        elif isinstance(other, (int, float)):
            # Add that constant to every element of A
            return Matrix([[A_row + other for A_row in rows] for rows in self.array])

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Create a new matrix
        # Check if the other object is of type Matrix
        if isinstance(other, Matrix):
            # Add the corresponding element of 1 matrices to another
            if other.size == 1:
                other = Matrix.fill(other.array[0][0], (self.rows, self.cols))
            return Matrix([[A_row - B_row for A_row, B_row in zip(*rows)] for rows in zip(self.array, other.array)])

                # If the other object is a scaler
        elif isinstance(other, (int, float)):
            # Add that constant to every element of A
            return Matrix([[A_row - other for A_row in rows] for rows in self.array])

    def __isub__(self, other):
        return self.__sub__(other)

        # Right addition can be done by calling left addition
    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):  # pointwise multiplication

        if isinstance(other, Matrix):
            if other.size == 1:
                other = Matrix.fill(other.array[0][0], (self.rows, self.cols))
            return Matrix([[A_row * B_row for A_row, B_row in zip(*rows)] for rows in zip(self.array, other.array)])

                # Scaler multiplication
        elif isinstance(other, (int, float)):
             return Matrix([[A_row * other for A_row in rows] for rows in self.array])

        # Point-wise multiplication is also commutative
    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):  # matrix-matrix multiplication
        if isinstance(other, Matrix):
            return Matrix([[sum(a * b for a, b in zip(A_row, B_col))
                        for B_col in zip(*other.array)]
                                for A_row in self.array])

    def __pow__(self, other):
        result = self.ones(dims=(self.rows, self.cols))

        if isinstance(other, (int, float)):
            for i in range(self.rows):
                for j in range(self.cols):
                    for times in range(other):
                        result.array[i][j] *= self.array[i][j] 

        return result
    
    def __truediv__(self, other):
        result = self.ones(dims=(self.rows, self.cols))# Scaler division

        if isinstance(other, (int, float)):
            for i in range(self.rows):
                for j in range(self.cols):
                    result.array[i][j] = self.array[i][j] / other
        
        return result

    def __getitem__(self, key):
        if isinstance(key, int):
            return Matrix([self.array[key]])
        else:
            return self.array[key[0]][key[1]]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            i = key[0]
            j = key[1]
            self.array[i][j] = value
    
    @staticmethod
    def fill(fill, dims):
        rows = dims[0]
        cols = dims[1]
        array = [[fill] * cols for i in range(rows)]
        return Matrix(array)
    
    @staticmethod
    def zeros(dims):
        return Matrix.fill(0, dims)

    @staticmethod
    def ones(dims):
        return Matrix.fill(1, dims)

    @staticmethod
    def random(dims):
        a = Matrix.zeros(dims)
        for i in range(dims[0]):
            for j in range(dims[1]):
                a[i,j] = random.extract_number()
        return a

    def mean(self):
        result = 0
        for i in range(self.rows):
            result += sum(self.array[i][:])
        return result/(self.cols*self.rows)
    
    def apply(self, function, inplace=False):
        out_array = Matrix.zeros((self.rows, self.cols))
        for i in range(self.rows):
                for j in range(self.cols):
                    out_array.array[i][j] = function(self.array[i][j])
        return out_array #if not inplace else new_array

    def T(self):
            return Matrix(list(map(list, zip(*self.array))))
