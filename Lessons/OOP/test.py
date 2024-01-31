from file import FileHandler

our_file = FileHandler()

our_file.write('new_file.txt', 'Hello World')

print('output', our_file.read('new_file.txt'))
                              