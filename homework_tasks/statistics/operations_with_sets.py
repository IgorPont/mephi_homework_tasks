"""
Представьте, что вы являетесь куратором образовательной программы, на которой будет два вебинара:
по программированию и по машинному обучению.

На вебинар по программированию записались потенциальные слушатели со следующими электронными адресами:

bennet@xyz.com
darcy@abc.com
margaret@xyz.com
pa@hhh.com
marimari@xyz.com
mallika@yahoo.com
abc@xyz.com
0071235@gmail.ru

На вебинар по машинному обучению записались потенциальные слушатели со следующими электронными адресами:

marimari@xyz.com
darcy@abc.com
0071235@gmail.ru
darcy@abc.com
petr44@xyz.com
katrin@ya.com

Оформите множества в Python для обоих списков слушателей.
"""

students_programming = {
    'bennet@xyz.com',
    'darcy@abc.com',
    'margaret@xyz.com',
    'pa@hhh.com',
    'marimari@xyz.com',
    'mallika@yahoo.com',
    'abc@xyz.com',
    '0071235@gmail.ru',
}

students_ml = {
    'marimari@xyz.com',
    'darcy@abc.com',
    '0071235@gmail.ru',
    'darcy@abc.com',
    'petr44@xyz.com',
    'katrin@ya.com',
}

# Задание 2.6
# С помощью операций множеств в Python определите, сколько слушателей записалось на оба вебинара.
# Пересечение множеств.
len_inter_students = len(students_programming & students_ml)

# Задание 2.7
# Сколько человек заинтересованы в посещении хотя бы одного вебинара?
# Объединение множеств.
len_union_students = len(students_programming | students_ml)

# Задание 2.8
# Сколько человек заинтересованы в посещении только одного вебинара из двух?
# Симметрическая разность.
len_symm_students = len(students_programming ^ students_ml)

if __name__ == "__main__":
    print(f'На оба вебинара записалось {len_inter_students} студентов.')
    print(f'Хотя бы одним семинаром заинтересовано {len_union_students} студентов.')
    print(f'Только одним семинаром заинтересовано {len_symm_students} студентов.')
