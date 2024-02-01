'''
Pseudo code:

request input from the user and store it into variable called 'name'
request input from the user and store it into variable called 'age'
request input from the user and store it into variable called 'house_number'
request input from the user and store it into variable called 'street'
output 'name', 'age', 'house_number' and 'street' in single sentence
'''

# Requests users name, age, house number and street name
name = input("Enter your name: ")
age =  input("How old are you? ")
house_number = input('What is your house number? ')
street = input('At what street you live in? ')

# Prints everything in single sentence
print("Dude's name is", name, "he is", age, "years old and lives at", house_number, street, '.')