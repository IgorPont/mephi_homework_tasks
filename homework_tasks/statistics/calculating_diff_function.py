import sympy as sp

"""
Вычисление производной функции первого порядка, второго порядка и в конкретной точке
"""

# 1️⃣ Объявляем переменную
x = sp.Symbol('x')

# 2️⃣ Задаем функцию
# f = 2 * (x ** 2) + 1
# f = sp.sin(3 * x) ** 2
# f = 2 * x**3 * sp.ln(x)
# f = x / (x**2 + 1)
# f = sp.sqrt(1 + x**2)
# f = (x**4) + (5*x)
f = x**3/4 - 3*x

# 3️⃣ Находим первую производную
f_prime = sp.diff(f, x)

# 4️⃣ Находим вторую производную
f_double_prime = sp.diff(f, x, 2)

# 5️⃣ Подставляем значение для конкретной точки
x_value = 4
f_prime_value = f_prime.subs(x, x_value)
f_double_prime_at_value = f_double_prime.subs(x, x_value)

# 6️⃣ Выводим результаты
print(f"f(x) = {f}")
print(f"f'(x) = {f_prime}")
print(f"f''(x) = {f_double_prime}")
print(f"f'(x_value) = {f_prime_value}")
print(f"f''(x_value) = {f_double_prime_at_value}")
