import sqlite3

conn = sqlite3.connect('database.db')  # Connect/create db
cursor = conn.cursor()  # we nned this to move arround


cursor.execute('''
    CREATE TABLE IF NOT EXISTS user(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name VARCHAR(255),
               grade INTEGER
    );
''')
conn.commit() # saves db

name = 'Jack'
grade = 80

cursor.execute('''
    CREATE TABLE user(
               INSERT INTO student(name, grade)
               VALUES (?,?)
    );
''', (name, grade))
conn.commit()

people = [
    ('Harry', 50),
    ('Tom', 70),
    ('Bob', 80),
]

# cursor.arraysize


# GET VALUES
cursor.execute('''
    SELECT * FROM student
''')
print(cursor.fetchall())