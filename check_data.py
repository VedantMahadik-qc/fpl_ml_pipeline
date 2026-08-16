import pandas as pd

# Load your combined database
df = pd.read_csv("fpl_combined_history.csv")

# Print out exactly how many match rows exist for each season
print("--- Rows per Season ---")
print(df['season'].value_counts())

# Check for a specific player (e.g., Bukayo Saka) to see his data
saka_data = df[df['player_name'].str.contains("Saka", na=False)]
print("\n--- Bukayo Saka's Most Recent Matches ---")
print(saka_data[['season', 'gameweek', 'team', 'total_points']].tail(5))