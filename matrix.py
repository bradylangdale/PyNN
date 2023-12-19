import random
import numpy as np


supported_constants = [int, float, np.float64]

class Matrix:

    def __init__(self, rows, cols, rand=True):
        self.rows = rows
        self.cols = cols

        self.data = (2 * (np.random.rand(rows, cols) - 0.5)) if rand else np.zeros(shape=(rows, cols))

    def transpose(self):
        result = Matrix(self.cols, self.rows, rand=False)
        result.data = self.data.transpose()
        return result
        
    def __str__(self):
        s = ''
        for r in self.data:
            s += str(r) + '\n'
        return s
    
    def __add__(self, b):
        if self.rows == b.rows and self.cols == b.cols:
            result = Matrix(self.rows, self.cols, rand=False)
            result.data = np.add(self.data, b.data)
            return result
        else:
            raise Exception('Matrixes A and B are not of the same size.')

    def __sub__(self, b):
        if self.rows == b.rows and self.cols == b.cols:
            result = Matrix(self.rows, self.cols, rand=False)
            result.data = np.subtract(self.data, b.data)
            return result
        else:
            if b.rows == 1 and b.cols == 1:
                result = Matrix(self.rows, self.cols, rand=False)
                result.data = np.subtract(self.data, b.data)
                return result
            else: 
                raise Exception('Matrixes A and B are not of the same size.')

    def __mul__(self, b):
        if type(b) in supported_constants:
            return self.__rmul__(b)
        elif self.cols == b.rows:
            return _multiply(self, b)
        else:
            raise Exception('Columns of matrix A does not match rows matrix B.')

    def __rmul__(self, const):
        if type(const) in supported_constants:
            result = Matrix(self.rows, self.cols, rand=False)
            result.data = self.data * const
            return result
        else:
            raise Exception('Can not multiply int and matrix together.')

    def __floordiv__(self, b):
        if type(b) in supported_constants:
            return self._div_(b)
        else:
            raise Exception('Operation not supported.')
    
    def __truediv__(self, b):
        if type(b) in supported_constants:
            return self._div_(b)
        else:
            raise Exception('Operation not supported.')        
    
    def _div_(self, const):
        if type(const) in supported_constants:
            result = Matrix(self.rows, self.cols, rand=False)
            result.data = self.data / const
            return result
        else:
            raise Exception('Can not divide int and matrix.')

    def __pow__(self, power):
        result = Matrix(self.rows, self.cols, rand=False)
        result.data = self.data

        for i in range(1, power):
            result = _multiply(result, self)

        return result
        
    def __getitem__(self, k):
        return self.data[k]
    
    def __setitem__(self, k, v):
        self.data[k] = v

    def __delitem__(self, k):
        del self.data[k]
        

def _multiply(a, b):
    result = Matrix(a.rows, b.cols, rand=False)
    result.data = np.matmul(a.data, b.data)
    return result

if __name__ == '__main__':
    print('m1')
    m1 = Matrix(2, 2)
    print(m1)

    print('m2')
    m2 = Matrix(2, 1)
    print(m2)

    print('m1 * m2')
    print(m1 * m2)

    try:
        print('m2 * m1')
        print(m2 * m1)
    except Exception as e:
        print('Error: ', e, end='\n\n')

    print('m1 + m1')
    print(m1 + m1)

    print(m1)
    print(m1**3)
    print(m1)

    print(2.0 * m1)
