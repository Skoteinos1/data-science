menu = ['coffee', 'tea', 'sugar', 'milk']
stock = {'coffee': 200, 'tea': 190, 'sugar': 220, 'milk': 321}
price = {'coffee': 5, 'tea': 4, 'sugar': 1, 'milk': 3}

total_stock = 0
for key in menu:
    total_stock += stock[key] * price[key]
    # print(key, price[key], stock[key], stock[key] * price[key])

print('Total Stock:', total_stock) 
