from math import sqrt
a = int(input("anna numero 1: "))
b = int(input("anna numero 2: "))
c = int(input("anna numero 3: "))

ans = (-b+sqrt(b**2-4*a*c))/(2*a)
ans2 = (-b-sqrt(b**2-4*a*c))/(2*a)
print(ans ,"ja", ans2)
