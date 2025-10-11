"""
Исследуемая функция: y = x**3 / (2 * (x + 5)**2)
"""
import sympy as sp

"""
Найдите область определения функции
"""
# Объявляем переменную
x = sp.Symbol('x')

# Задаем функцию
y = x ** 3 / (2 * (x + 5) ** 2)

# 1️⃣ Область определения
domain = sp.calculus.util.continuous_domain(y, x, sp.S.Reals)

# 3️⃣ Выводим результаты
print("Функция:", y)
print("Область определения:", domain)

"""
Найдите область значений функции

import sympy as sp

x = sp.Symbol('x', real=True)
f = x**3 / (2*(x + 5)**2)

# Область значений (диапазон функции)
range_f = sp.calculus.util.function_range(f, x, sp.S.Reals)
print(range_f)
"""
# res = sp.calculus.util.function_range(y, x, sp.S.Reals)
# {x**3/(2*x**2 + 20*x + 50)} — это "задано через формулу", а не готовый интервал.
# Чтобы получить явный интервал, посчитаем диапазон на двух интервалах непрерывности и объединим.

x = sp.Symbol('x', real=True)
g = x ** 3 / (2 * (x + 5) ** 2)
# считаем по частям из-за разрыва в x = -5
rng_left = sp.calculus.util.function_range(g, x, sp.Interval.open(-sp.oo, -5))
rng_right = sp.calculus.util.function_range(g, x, sp.Interval.open(-5, sp.oo))

rng = sp.simplify(rng_left.union(rng_right))
print("Область значений:", rng)  # -> (-oo, oo)

"""
Исследуйте функцию на чётность
"""
x = sp.symbols('x', real=True)
f = x ** 3 / (2 * (x + 5) ** 2)

print(sp.simplify(f.subs(x, -x) - f))  # != 0 → нечетная? нет
print(sp.simplify(f.subs(x, -x) + f))  # != 0 → четная?  нет

"""
В какой точке график пересекает ось абсцисс?
"""
x = sp.Symbol('x')
q = x ** 3 / (2 * (x + 5) ** 2)
x_intercept = sp.solve(sp.Eq(q, 0), x)
print("Пересекает ось абсцисс:", x_intercept)

"""
В какой точке график пересекает ось ординат?
"""
x = sp.Symbol('x')
h = x ** 3 / (2 * (x + 5) ** 2)
h_at_0 = h.subs(x, 0)
print("Пересекает ось ординат:", (0, h_at_0))

"""
Найдите производную от функции
"""
x = sp.Symbol('x')
a = x ** 3 / (2 * (x + 5) ** 2)
y_prime = sp.diff(a, x)
print("Производная функции:", y_prime)  # -> x**2*(x + 15)/(2*(x + 5)**3)

"""
Найдите точку максимума
"""
x = sp.symbols('x', real=True)
y = x ** 3 / (2 * (x + 5) ** 2)

y1 = sp.diff(y, x)
y2 = sp.diff(y1, x)

crit = sp.solve(sp.Eq(y1, 0), x)  # [-15, 0]

print("y'(x) =", sp.simplify(y1))
print("y''(x) =", sp.simplify(y2))
print("Точка максимума:", crit)

"""
Найдите точку минимума.

Теперь смотрим на поведение функции:
    - При x=−15: производная меняет знак с + на – -> максимум.
    - При x=0: производная не меняет знак -> точка перегиба, не экстремум.
    - При x=−5: производная не существует (деление на ноль), и это разрыв функции.
    
Области возрастания функции:
    (-oo; -15), (-5; 0), (0; +oo)
    
Интервалы выпуклости функции:
    (-oo; -5), (-5; 0)

Таким образом точки минимума нет. 
"""

"""
Полный код для построения графика
"""
# import sympy as sp
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Определяем переменную и функцию
# x = sp.Symbol('x', real=True)
# y = x ** 3 / (2 * (x + 5) ** 2)
#
# # Находим критические точки
# y_prime = sp.diff(y, x)
# critical_points = sp.solve(sp.Eq(y_prime, 0), x)  # [-15, 0]
#
# # Преобразуем в функцию для numpy
# f = sp.lambdify(x, y, 'numpy')
#
# # Диапазон x (исключаем разрыв в -5)
# x_vals_left = np.linspace(-40, -5.1, 400)
# x_vals_right = np.linspace(-4.9, 20, 400)
#
# # Вычисляем значения
# y_vals_left = f(x_vals_left)
# y_vals_right = f(x_vals_right)
#
# # График
# plt.figure(figsize=(9, 6))
# plt.plot(x_vals_left, y_vals_left, label='y = x³ / (2·(x+5)²)', color='royalblue')
# plt.plot(x_vals_right, y_vals_right, color='royalblue')
#
# # Отмечаем критические точки
# for c in critical_points:
#     plt.scatter(float(c), float(y.subs(x, c)), color='red', zorder=5)
#     plt.text(float(c) + 0.5, float(y.subs(x, c)) + 5,
#              f'x={float(c)}', color='red', fontsize=10)
#
# # Вертикальная линия x = -5 (разрыв)
# plt.axvline(-5, color='gray', linestyle='--', label='x = -5 (разрыв)')
#
# plt.title("Исследование функции y = x³ / (2(x+5)²)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid(True)
# plt.legend()
# plt.show()
