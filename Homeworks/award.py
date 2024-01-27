a = int(input('How long it took to complete swimming? '))
b = int(input('How long it took to complete cycling? '))
c = int(input('How long it took to complete running? '))

print('Total time taken to complete the triathlon: ', a+b+c, 'minutes')
if a+b+c <= 100:
    print("Your Award is: Provincial Colours")
elif a+b+c <= 105:
    print("Your Award is: Provincial Half Colours")
elif a+b+c <= 110:
    print("Your Award is: Provincial Scroll")
else:
    print("No soup for you.")