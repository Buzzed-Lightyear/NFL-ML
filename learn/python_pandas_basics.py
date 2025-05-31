import pandas as pd
import numpy as np

# Python Fundamentals Examples
print("=== Python Fundamentals ===")

# 1. Basic Data Types
integer_example = 42
float_example = 3.14
string_example = "Hello, NFL!"
boolean_example = True

print(f"Integer: {integer_example}")
print(f"Float: {float_example}")
print(f"String: {string_example}")
print(f"Boolean: {boolean_example}")

# 2. Lists and Dictionaries
# List example
teams = ["Chiefs", "Eagles", "49ers", "Ravens"]
print("\nList of teams:", teams)
print("First team:", teams[0])
print("Last team:", teams[-1])

# Dictionary example
team_stats = {
    "Chiefs": {"wins": 11, "losses": 6},
    "Eagles": {"wins": 11, "losses": 6},
    "49ers": {"wins": 12, "losses": 5},
    "Ravens": {"wins": 13, "losses": 4}
}
print("\nTeam stats dictionary:", team_stats)
print("Chiefs wins:", team_stats["Chiefs"]["wins"])

# Pandas Examples
print("\n=== Pandas Examples ===")

# Create a sample DataFrame
data = {
    'Team': ['Chiefs', 'Eagles', '49ers', 'Ravens'],
    'Wins': [11, 11, 12, 13],
    'Losses': [6, 6, 5, 4],
    'Points_For': [371, 433, 491, 483],
    'Points_Against': [294, 428, 298, 280]
}

# Create DataFrame
df = pd.DataFrame(data)

# Basic DataFrame operations
print("\n1. Display first few rows:")
print(df.head())

print("\n2. DataFrame information:")
print(df.info())

print("\n3. Basic statistics:")
print(df.describe())

# Column operations
print("\n4. Select specific columns:")
print(df[['Team', 'Wins']])

# Filtering
print("\n5. Teams with more than 11 wins:")
print(df[df['Wins'] > 11])

# Adding a new column
df['Win_Percentage'] = df['Wins'] / (df['Wins'] + df['Losses'])
print("\n6. Added win percentage column:")
print(df)

# Sorting
print("\n7. Sort by wins (descending):")
print(df.sort_values('Wins', ascending=False))

# Basic calculations
print("\n8. Average points scored per team:")
print(df['Points_For'].mean())

print("\n9. Total points scored by all teams:")
print(df['Points_For'].sum())

# Save DataFrame to CSV
df.to_csv('sample_nfl_stats.csv', index=False)
print("\nSaved sample data to 'sample_nfl_stats.csv'") 