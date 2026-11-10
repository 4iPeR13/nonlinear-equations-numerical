import numpy as np
from sympy import symbols, Poly, div, simplify, sign, lambdify

x = symbols('x')
coeffs = [2, -2, -4, 0, 2, 1]
poly = Poly(coeffs, x)

a0 = coeffs[-1]
an = coeffs[0]
A = max(abs(c) for c in coeffs[:-1])
B = max(abs(c) for c in coeffs[1:])
left = abs(a0) / (abs(a0) + A)
right = 1 + B / abs(an)
print(f"Корені лежать у кільці: [{left:.4f}, {right:.4f}]")

# Теорема Гюа
def gua_theorem(coeffs):
    for k in range(1, len(coeffs) - 1):
        if coeffs[k] < 0 and coeffs[k]**2 < coeffs[k - 1] * coeffs[k + 1]:
            return True
    return False

if gua_theorem(coeffs):
    print("Рівняння має хоча б одну пару комплексноспряжених коренів.")
else:
    print("Рівняння не має жодної пари комплексноспряжених коренів.")

# Послідовність Штурма
def sturm_sequence(p):
    seq = [p, p.diff()]
    while True:
        q, r = div(seq[-2], seq[-1], domain='QQ')
        if r == 0:
            break
        seq.append(-simplify(r))
    return seq

def sign_changes(seq, val):
    values = [s.evalf(subs={x: val}) for s in seq]
    signs = [sign(v) for v in values if v != 0]
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])

seq = sturm_sequence(poly.as_expr())
Va = sign_changes(seq, -10)
Vb = sign_changes(seq, 10)
print(f"Кількість дійсних коренів на [-10, 10]: {Va - Vb}")


intervals = []
seq = sturm_sequence(poly.as_expr())

step = 0.5
for i in range(int(20 / step)):
    a = -10 + i * step
    b = a + step
    Va = sign_changes(seq, a)
    Vb = sign_changes(seq, b)
    if Va - Vb > 0:
        intervals.append((a, b))
        print(f"Має корень на [{a}, {b}]")

f = lambdify(x, poly.as_expr(), 'numpy')

# Метод бісекції

def bisection(f, a, b, eps=1e-6):
    steps = 0
    while abs(b - a) > eps:
        c = (a + b) / 2
        if f(a) * f(c) <= 0:
            b = c
        else:
            a = c
        steps += 1
        print(f"Ітерація {steps}: x = {c:.6f}, |b - a| = {abs(b - a):.6e}")
    return (a + b) / 2, steps

# Метод хорд
def chord(f, a, b, eps=1e-6):
    steps = 0
    x0, x1 = a, b
    while abs(x1 - x0) > eps:
        x_next = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x_next
        steps += 1
        print(f"Ітерація {steps} (Хорд): x = {x1:.6f}, |x1 - x0| = {abs(x1 - x0):.6e}")
    return x1, steps
f_prime = lambdify(x, poly.diff().as_expr(), 'numpy')
# Метод Ньютона
def newton(f, f_prime, x0, eps=1e-6, max_iter=1000):
    steps = 0
    while abs(f(x0)) > eps and steps < max_iter:
        x0 = x0 - f(x0) / f_prime(x0)
        steps += 1
        print(f"Ітерація {steps} (Ньютон): x = {x0:.6f}, |f(x)| = {abs(f(x0)):.6e}")
    return x0, steps

# Обчислення коренів і порівняння
for i, (a, b) in enumerate(intervals):
    print(f"\nКорінь {i+1} на [{a:.2f}, {b:.2f}]:")
    r_bis, s_bis = bisection(f, a, b)
    r_ch, s_ch = chord(f, a, b)
    r_newt, s_newt = newton(f, f_prime, (a + b) / 2)
    print(f"Бісекція: x = {r_bis:.6f}, ітерацій: {s_bis}")
    print(f"Хорд:     x = {r_ch:.6f}, ітерацій: {s_ch}")
    print(f"Ньютон:   x = {r_newt:.6f}, ітерацій: {s_newt}")