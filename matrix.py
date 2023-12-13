import random


class Matrix:

    def __init__(self, rows, cols, rand=True):
        self.rows = rows
        self.cols = cols

        self.data = [[random.uniform(-1, 1) if rand else 0 for c in range(self.cols)] 
            for r in range(self.rows)]
        
    def __str__(self):
        s = ''
        for r in self.data:
            s += str(r) + '\n'
        return s
    
    def __add__(self, b):
        if self.rows == b.rows and self.cols == b.cols:
            result = Matrix(self.rows, self.cols, rand=False)
            
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i][j] = self[i][j] + b[i][j]
            
            return result
        else:
            raise Exception('Matrixes A and B are not of the same size.')

    def __sub__(self, b):
        if self.rows == b.rows and self.cols == b.cols:
            result = Matrix(self.rows, self.cols, rand=False)
            
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i][j] = self[i][j] - b[i][j]
            
            return result
        else:
            raise Exception('Matrixes A and B are not of the same size.')

    def __mul__(self, b):
        if self.cols == b.rows:
            return _multiply(self, b)
        else:
            raise Exception('Columns of matrix A does not match rows matrix B.')

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

    for i in range(a.rows):
        for j in range(b.cols):
            entry = 0
            for k in range(a.cols):
                entry += a[i][k] * b[k][j]
            result[i][j] = entry
    
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
