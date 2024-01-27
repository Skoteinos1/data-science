# Write a program with two compilation errors, a runtime error and a logical error.

x = 0
if x == 0  # compilation error: SyntaxError: missing :
print(x)  # Compilation error: IndentationError:  command shouldn't start at beginning of line
    print(3/x)   # Runtime error: division by zero

a = 1
b = 2
area_of_rectangle = a/b  # LogicalError: Area of Rectangle is calculated as a*b