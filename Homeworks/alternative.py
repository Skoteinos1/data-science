# Write a program that reads in a string and makes each alternate character into an upper case character and each other alternate character a lower case character.
# Sorry no idea what you mean by: "reads in a string" - Do you want it to be read from a file? As input by user? So I choose simplest method and set it directly

base_string = 'Dog jumps over lazy fox.'

# alternate character into an upper case character and each other alternate character a lower case
s1 = ''
for i in range(len(base_string)):
    if i % 2 == 0:
        s1 += base_string[i].upper()
    else:
        s1 += base_string[i].lower()
print(s1)

# each alternative word lower and upper case
base_string = base_string.split(' ')
for i in range(len(base_string)):
    if i % 2 == 0:
        base_string[i] = base_string[i].lower()
    else:
        base_string[i] = base_string[i].upper()
print(" ".join(base_string))