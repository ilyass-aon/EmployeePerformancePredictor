CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT CHECK(role IN ('admin', 'manager', 'employee')) NOT NULL
);

CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    name TEXT,
    department TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS performance_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER,
    hours_worked REAL,
    tasks_completed INTEGER,
    feedback_score REAL,
    punctuality_rate REAL,
    performance_score REAL,
    date TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(id)
);

ALTER TABLE employees ADD COLUMN manager_id INTEGER;
ALTER TABLE performance_data ADD COLUMN manager_id INTEGER;
