import sqlite3

def create_student(name:str, grade:int):
    conn = sqlite3.connect('database.db')
    # ...
    conn.close()


conn = sqlite3.connect('database.db')  # Connect/create db
cursor = conn.cursor()  # we nned this to move arround




cursor.execute('''
    CREATE TABLE IF NOT EXISTS student(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name VARCHAR(255),
               grade INTEGER
    );
''')
conn.commit() # saves db

name = 'Jack'
grade = 80

cursor.execute('''
               INSERT INTO student(name, grade)
               VALUES (?,?)
''', (name, grade))
conn.commit()

people = [
    ('Harry', 50),
    ('Tom', 70),
    ('Bob', 80),
]

cursor.execute('''
''')


# GET VALUES
cursor.execute('''
    SELECT * FROM student
    WHERE name = 'Jack'
''')
print(cursor.fetchall())
print(cursor.fetchone())

# UPDATE
cursor.execute('''
               UPDATE student
               SET grade = 40

''')




conn.close()  # Always close connection