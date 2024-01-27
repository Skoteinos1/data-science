# This example program is meant to demonstrate errors.
 
# There are some errors in this program. Run the program, look at the error messages, and find and fix the errors.

print("Welcome to the error program")  # SyntaxError: Missing parentheses
print("\n") # IndentationError: unexpected indent  &  SyntaxError: Missing parentheses

    # Variables declaring the user's age, casting the str to an int, and printing the result   
age_Str = "24"  # IndentationError: unexpected indent  &  SyntaxError: there should be only one =  & Logical error: there shouldn't be ' years old'
age = int(age_Str)  # IndentationError: unexpected indent
print("I'm" + str(age) + "years old.")  # IndentationError: unexpected indent  &  TypeError: you try to use integer as string

    # Variables declaring additional years and printing the total years of age
years_from_now = "3"  # IndentationError: unexpected indent
total_years = age + int(years_from_now)  # IndentationError: unexpected indent  &  TypeError: you try to add string and integer

print("The total number of years:" + str(total_years))  # SyntaxError: Missing parentheses  &  SyntaxError: wrong variable name

# Variable to calculate the total amount of months from the total amount of years and printing the result
total_months = total_years * 12  # SyntaxError: wrong variable name
print("In 3 years and 6 months, I'll be " + str(total_months+6) + " months old")  # SyntaxError: Missing parentheses  &  TypeError: you try to use integer as string  &  LogicError: you forgot add 6

#HINT, 330 months is the correct answer

