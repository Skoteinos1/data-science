import math

msg = '''investment - to calculate the amount of interest you'll earn on your investment
bond - to calculate the amount you'll have to pay on a home loan
Enter either 'investment' or 'bond' from the menu above to proceed:'''
print(msg)

choice = input()
choice = choice.lower()

all_fine = False
if choice == 'investment':
    p = float(input('What is the amount of money that you are depositing: '))
    r = float(input('What is the interest rate: '))
    t = float(input('How long you plan to invest: '))
    interest = input('“simple” or “compound” interest: ').lower()
    if interest == 'simple':
        A = p *(1 + r*t/100)
    elif interest == 'compound':
        A = p * (1+r/100)**t
    print("You will have this much money:", A)
    all_fine = True

elif choice == 'bond':
    p = float(input('What is the present value of the house: '))
    i = float(input('What is the interest rate: '))/1200
    n = float(input('Number of months you plan to spend repaying the bond: '))
    repayment = (i * p)/(1 - (1 + i)**(-n))
    print('Every month you will have to repay:', round(repayment,2))
    all_fine = True

if not all_fine:
    print('Cat like typing detected. Meeooowww')