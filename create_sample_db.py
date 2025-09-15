#!/usr/bin/env python3
"""
Script to create a sample SQLite database for the kilter-board-predictor project.
This creates a realistic climbing route database with holds, routes, and user attempts.
"""

import sqlite3
import random
from datetime import datetime, timedelta

def create_sample_database():
    # Connect to SQLite database
    conn = sqlite3.connect('kilter_board_data.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS holds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position_x REAL NOT NULL,
            position_y REAL NOT NULL,
            hold_type TEXT NOT NULL,
            difficulty_contribution INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS routes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            grade TEXT NOT NULL,
            grade_numeric INTEGER NOT NULL,
            setter_name TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS route_holds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            route_id INTEGER NOT NULL,
            hold_id INTEGER NOT NULL,
            hold_role TEXT NOT NULL, -- 'start', 'foot', 'hand', 'finish'
            FOREIGN KEY (route_id) REFERENCES routes (id),
            FOREIGN KEY (hold_id) REFERENCES holds (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            route_id INTEGER NOT NULL,
            attempt_result TEXT NOT NULL, -- 'flash', 'send', 'attempt', 'fail'
            attempts_count INTEGER DEFAULT 1,
            user_grade_estimate TEXT,
            difficulty_rating INTEGER, -- 1-10 scale
            attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (route_id) REFERENCES routes (id)
        )
    ''')
    
    # Insert sample holds (representing a kilter board layout)
    holds_data = []
    for i in range(200):  # 200 holds on the board
        x = random.uniform(0, 100)  # board width percentage
        y = random.uniform(0, 100)  # board height percentage
        hold_type = random.choice(['jug', 'crimp', 'pinch', 'sloper', 'pocket'])
        difficulty = random.randint(1, 10)
        holds_data.append((x, y, hold_type, difficulty))
    
    cursor.executemany('INSERT INTO holds (position_x, position_y, hold_type, difficulty_contribution) VALUES (?, ?, ?, ?)', holds_data)
    
    # Insert sample routes
    grades = ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
    grade_numeric = {grade: idx for idx, grade in enumerate(grades)}
    setters = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
    
    routes_data = []
    for i in range(50):  # 50 routes
        grade = random.choice(grades)
        name = f"Route {i+1}"
        setter = random.choice(setters)
        description = f"A challenging {grade} route set by {setter}"
        routes_data.append((name, grade, grade_numeric[grade], setter, description))
    
    cursor.executemany('INSERT INTO routes (name, grade, grade_numeric, setter_name, description) VALUES (?, ?, ?, ?, ?)', routes_data)
    
    # Insert route-hold relationships
    route_holds_data = []
    for route_id in range(1, 51):  # for each route
        # Each route uses 8-15 holds
        num_holds = random.randint(8, 15)
        selected_holds = random.sample(range(1, 201), num_holds)
        
        # Assign roles to holds
        start_holds = selected_holds[:2]  # 2 start holds
        finish_holds = selected_holds[-2:]  # 2 finish holds
        middle_holds = selected_holds[2:-2]  # remaining holds
        
        for hold_id in start_holds:
            route_holds_data.append((route_id, hold_id, 'start'))
        
        for hold_id in finish_holds:
            route_holds_data.append((route_id, hold_id, 'finish'))
        
        for hold_id in middle_holds:
            role = random.choice(['hand', 'foot'])
            route_holds_data.append((route_id, hold_id, role))
    
    cursor.executemany('INSERT INTO route_holds (route_id, hold_id, hold_role) VALUES (?, ?, ?)', route_holds_data)
    
    # Insert user attempts
    attempts_data = []
    for i in range(1000):  # 1000 attempts
        user_id = random.randint(1, 20)  # 20 users
        route_id = random.randint(1, 50)
        result = random.choice(['flash', 'send', 'attempt', 'fail'])
        attempts = random.randint(1, 20) if result != 'flash' else 1
        grade_estimate = random.choice(grades)
        difficulty_rating = random.randint(1, 10)
        
        # Create random timestamp within the last year
        base_date = datetime.now() - timedelta(days=365)
        random_days = random.randint(0, 365)
        attempt_date = base_date + timedelta(days=random_days)
        
        attempts_data.append((user_id, route_id, result, attempts, grade_estimate, difficulty_rating, attempt_date))
    
    cursor.executemany('INSERT INTO user_attempts (user_id, route_id, attempt_result, attempts_count, user_grade_estimate, difficulty_rating, attempted_at) VALUES (?, ?, ?, ?, ?, ?, ?)', attempts_data)
    
    # Commit and close
    conn.commit()
    conn.close()
    print("Sample database 'kilter_board_data.db' created successfully!")
    print("Database contains:")
    print("- 200 holds")
    print("- 50 routes")
    print("- Route-hold relationships")
    print("- 1000 user attempts")

if __name__ == "__main__":
    create_sample_database()