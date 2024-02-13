import sqlite3

def create_student(name:str, grade:int):
    conn = sqlite3.connect('database.db')
    # ...
    conn.close()


conn = sqlite3.connect('database.db')  # Connect/create db
cursor = conn.cursor()  # we nned this to move arround

# CREATE TABLE
cursor.execute('''
    CREATE TABLE IF NOT EXISTS student(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name VARCHAR(255),
               grade INTEGER
    );
''')
conn.commit() # saves db

# Add one
name = 'Jack'
grade = 80
cursor.execute('''
               INSERT INTO student(name, grade)
               VALUES (?,?)
''', (name, grade))
conn.commit()

# Add multiple
people = [
    ('Harry', 50),
    ('Tom', 70),
    ('Bob', 80),
]
cursor.executemany('''
               INSERT INTO student(name, grade)
               VALUES (?,?)
''', people)
conn.commit()


cursor.execute('''
''')


# GET VALUES
cursor.execute('''
    SELECT * FROM student
    WHERE name = 'Jack'
''')
print(cursor.fetchall())

id = 3
cursor.execute("SELECT * FROM student WHERE id = ?", (id,))
print(cursor.fetchone())

# UPDATE
grade = 85
id = 1
cursor.execute('''
               UPDATE student
               SET grade = ?
               WHERE id = ?
               ''', (grade, id))


id = 2
cursor.execute("DELETE FROM student WHERE id = ?", (id,))
conn.commit()

# SQL Injection protection --------------
# This function passes the data correctly 
def get_tasks_correctly(user_id: int):
    with sqlite3.connect('task.db') as db:
        cursor = db.cursor()

        cursor.execute('''
            SELECT user_id, title, description
            FROM tasks
            WHERE user_id = ?
        ''', (user_id,))

        return cursor.fetchall()
    

# Get the tasks incorrectlty 
def get_tasks_incorrectly(user_id: id):
    with sqlite3.connect('task.db') as db:
        cursor = db.cursor()

        cursor.execute(f'''
            SELECT user_id, title, description
            FROM tasks
            WHERE user_id = {user_id}
        ''')

        return cursor.fetchall()

print('Using Correct function:')
print(*get_tasks_correctly(2), sep='\n')

print('\nUsing the incorrect function:')
print(*get_tasks_incorrectly(1), sep='\n')

# Injection script 
injection_script = '''
    1
    UNION 
    SELECT id, password, email
    FROM user 
'''

print('Using Correct function:')
print(*get_tasks_correctly(injection_script), sep='\n')

print('\nUsing the incorrect function:')
print(*get_tasks_incorrectly(injection_script), sep='\n')

conn.close()  # Always close connection


# ---------- Connecting to DB in Python --------------
import pyodbc

conn_string = '''Driver={SQL Server Native Client 11.0};
                 Server=(localdb)\MSSQLLocalDB;
                 Database=ACME_DATABASE;
                 Trusted_Connection=yes;'''

conn = pyodbc.connect(conn_string)

cursor = conn.cursor()
cursor.execute('SELECT TOP(5) * FROM product')

for row in cursor:
    print(row)

conn.close()

# ---------- Connecting to Pandas --------------
import pyodbc
import pandas as pd

conn_string = '''Driver={SQL Server Native Client 11.0};
                 Server=(localdb)\MSSQLLocalDB;
                 Database=ACME_DATABASE;
                 Trusted_Connection=yes;'''

conn = pyodbc.connect(conn_string)

sql_query = """
    SELECT * 
    FROM transactions    
"""

transactions_df = pd.read_sql(sql_query, conn)
transactions_df.tail()

# -
sql_query = """
    SELECT TransactionDate as date, 
	   PricePaid as price, ProdName as product,
	   CategoryName as category
    FROM Transactions t
    LEFT JOIN Product p
        ON t.ProductID = p.ProductID
    LEFT JOIN Category c
        ON t.CategoryID = c.CategoryID
"""

transactions_df = pd.read_sql(sql_query, conn)
transactions_df.tail()

# - 
import matplotlib.pyplot as plt
total_sale_prices_by_date = transactions_df.groupby("date")["price"].sum()
total_sales_by_date = transactions_df.groupby("date")["price"].count()  

fig, ax = plt.subplots()

ax.plot(total_sale_prices_by_date.index, total_sales_by_date.values, color="b")
ax.bar(total_sales_by_date.index, total_sales_by_date.values, color='r')
ax.set_xlabel('Date')
plt.xticks(rotation=90)

plt.show()


# ---------- Connecting to ORM --------------

from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Integer, String, CHAR, Date
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import sessionmaker

class Base(DeclarativeBase):
    pass

class Person(Base):
    __tablename__ = 'person' # Required, name of the table
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    fullname = Column(String(50))
    nickname = Column(String(50))

    def __init__(self, name, fullname, nickname):
        self.name = name
        self.fullname = fullname
        self.nickname = nickname

    def __repr__(self):
        return f"<Person(name={self.name}, fullname={self.fullname}, nickname={self.nickname})>"
    

engine = create_engine('sqlite:///people.db', echo=True) # The database we want to use (sqlite) and th e name of the file (people.db)
Base.metadata.create_all(bind=engine) # connects all of the databases that connect to the base 

Session = sessionmaker(bind=engine)
session = Session()

person = Person('Jack', 'Jack Smith', 'Jackie') # Crate our person object
session.add(person) # Write the person to the database
session.commit() # Commit the changes to the database


people = [
    Person('Matthew', 'Matthew Sanders ', 'M. Shadows'),
    Person('James', 'James Sullivan', 'The Rev'),
    Person('Brian', 'Brian Haner ', 'Synyster Gates'),
    Person('Zachary', 'Zachary Baker ', 'Synyster Gates'),
    Person('Jonathan', 'Jonathan Seward', 'Johnny Christ'),
]

for person in people:
    session.add(person)

session.commit()



all_people = session.query(Person).all()
print(all_people)


the_rev = session.query(Person).where(Person.name == 'James').first()
print("Name:", the_rev.name)
print("Full Name:", the_rev.fullname)
print("Nickname:", the_rev.nickname)




