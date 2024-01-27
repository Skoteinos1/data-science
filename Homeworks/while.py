i = 0
sum = 0
c = 0
while i != -1:
    i = int(input('Enter Number (-1 for exit): '))
    if i != -1:
        sum += i
        c += 1
print('Average of entered numbers without -1 is:', sum/c)
