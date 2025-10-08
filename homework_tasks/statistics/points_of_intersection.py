"""
1. Найдите координаты пересечения с осью x для функции y = x**2 + 2*x - 8
"""

from sympy import symbols, Eq, solve

# Сначала находим точку пересечения с осью x
x = symbols('x')
equation = Eq(x ** 2 + 2 * x - 8, 0)
solutions = solve(equation)
print(solutions)

# Теперь точку пересечения с y
x = symbols('x')
y = x**2 + 2*x - 8
y_value = y.subs(x, 0)
print((0, y_value))

"""
2. Найдите область значений для функции f(x) = 3 / (x**2 - 10)
"""
from sympy import symbols
from sympy.calculus.util import function_range
from sympy import S

x = symbols('x')
f = 3 / (x**2 - 10)

# Найдём область значений
range_f = function_range(f, x, S.Reals)
print(range_f)