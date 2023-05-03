# class IntRoot5:
#     def __init__(self, integer, root5):
#         self.integer = integer
#         self.root5 = root5
    
#     def __sub__(self, other):
#         return IntRoot5(self.integer - other.integer, self.root5 - other.root5)

#     def __add__(self, other):
#         return IntRoot5(self.integer + other.integer, self.root5 + other.root5)

#     def __mul__(self, other):
#         new_integer = self.integer * other.integer + 5 * self.root5 * other.root5
#         new_root5 = self.integer * other.root5 + self.root5 * other.integer
#         return IntRoot5(new_integer, new_root5)

#     def __pow__(self, n):
#         if n == 0:
#             return IntRoot5(1, 0)
        
#         elif n % 2 == 0:
#             temp = self ** (n // 2)
#             return temp * temp
        
#         else:
#             return self * (self ** (n - 1))

#     def __repr__(self):
#         return f"{self.integer} + {self.root5} * sqrt(5)"


# def binet_formula(n):
#     phi = IntRoot5(1, 1)  # (1 + sqrt(5)) / 2
#     psi = IntRoot5(1, -1)  # (1 - sqrt(5)) / 2

#     result = (phi ** n - psi ** n).root5//2**n

#     return result

# n=100000
# print(f"F({n}) = {binet_formula(n)}")

def matrix_multiply(a, b):
    result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result

def power_matrix(a,n):
    if n==1:
        return a
    elif n%2==0:
        return matrix_multiply(a, a)
    else:
        return matrix_multiply(power_matrix(a,n//2),power_matrix(a+1,n//2))

def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        matrix = [[1, 1], [1, 0]]
        initial_vector = [1, 0]
        matrix_n_minus_1 = power_matrix(matrix, n - 1)
        fn = matrix_n_minus_1[0][0] * initial_vector[0] + matrix_n_minus_1[0][1] * initial_vector[1]
        return fn
