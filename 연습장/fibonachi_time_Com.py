import math

operation_count = 0

class IntRoot5:
    def __init__(self, integer, root5):
        self.integer = integer
        self.root5 = root5

    def __sub__(self, other):
        global operation_count
        operation_count += 1
        return IntRoot5(self.integer - other.integer, self.root5 - other.root5)

    def __add__(self, other):
        global operation_count
        operation_count += 1
        return IntRoot5(self.integer + other.integer, self.root5 + other.root5)

    def __mul__(self, other):
        global operation_count
        operation_count += 2
        new_integer = self.integer * other.integer + 5 * self.root5 * other.root5
        new_root5 = self.integer * other.root5 + self.root5 * other.integer
        return IntRoot5(new_integer, new_root5)

    def __pow__(self, n):
        if n == 0:
            return IntRoot5(1, 0)

        elif n % 2 == 0:
            temp = self ** (n // 2)
            return temp * temp

        else:
            return self * (self ** (n - 1))

    def __repr__(self):
        return f"{self.integer} + {self.root5} * sqrt(5)"


def binet_formula(n):
    global operation_count
    phi = IntRoot5(1, 1)  # (1 + sqrt(5)) / 2
    psi = IntRoot5(1, -1)  # (1 - sqrt(5)) / 2

    result = (phi ** n - psi ** n).root5 // (2 ** n)
    operation_count += int(math.ceil(math.log2(n)))  # 2 ** n 계산에 필요한 연산 횟수

    return result

n = 100000000000
operation_count = 0
print(f"F({n}) = {binet_formula(n)}")
print(f"Operation count: {operation_count}")