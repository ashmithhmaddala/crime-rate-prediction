import sqlite3

# Connect to database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table with proper columns
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    role TEXT NOT NULL
)
''')

# Insert a test admin and user
c.execute("INSERT OR IGNORE INTO users (email, password, role) VALUES (?, ?, ?)", ('admin@osmo.com', 'admin123', 'admin'))
c.execute("INSERT OR IGNORE INTO users (email, password, role) VALUES (?, ?, ?)", ('user@osmo.com', 'user123', 'user'))

# Save and close
conn.commit()
conn.close()

print("✅ Database initialized with users table.")
