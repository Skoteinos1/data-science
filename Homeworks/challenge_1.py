a = int(input('Enter length of first side of triangle '))
b = int(input('Enter length of second side of triangle '))
c = int(input('Enter length of third side of triangle '))

s = (a + b + c)/2
area = (s*(s-a)*(s-b)*(s-c))**0.5
print('Area of triangle is', area)