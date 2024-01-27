"""Ask the user to enter three different integers.
● Then print out:
○ The sum of all the numbers
○ The first number minus the second number
○ The third number multiplied by the first number
○ The sum of all three numbers divided by the third number"""

a = int(input('Enter first integer '))
b = int(input('Enter second integer '))
c = int(input('Enter third integer '))

print('The sum of all the numbers', a+b+c)
print('The first number minus the second number', a-b)
print('The third number multiplied by the first number', c*a)
print('The sum of all three numbers divided by the third number', (a+b+c)/c)
