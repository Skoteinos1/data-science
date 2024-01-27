string_fav = input('Enter your favourite restaurant ')
int_fav = input('Enter your favourite number ')

print('Your Restaurant', string_fav)
print('Your Number    ', int_fav)
try:
    # This will not work obviously. Word is not a number.
    # But hey, maybe I am wrong. Just for sake of argument, can you convert "Pizza Hut" into number, please?
    int(string_fav)
except:
    pass