import os
import time
import pandas as pd
from nba_api.stats.endpoints import playergamelogs

CSV_PATH = "game_logs.csv"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]


def load_logs():
    # Reuse the saved file so you don't re-download on every run
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)

    all_logs = []
    for season in SEASONS:
        print(f"Pulling {season}...")
        df = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
        df["SEASON"] = season
        all_logs.append(df)
        time.sleep(2)  # avoid rate limits

    logs = pd.concat(all_logs, ignore_index=True)
    logs.to_csv(CSV_PATH, index=False)
    return logs


def add_fantasy_points(logs):
    # Missed shots
    logs["FGMI"] = logs["FGA"] - logs["FGM"]
    logs["FTMI"] = logs["FTA"] - logs["FTM"]

    # Double doubles and triple doubles
    cats = logs[["PTS", "REB", "AST", "STL", "BLK"]]
    num_tens = (cats >= 10).sum(axis=1)
    logs["DD"] = (num_tens >= 2).astype(int)
    logs["TD"] = (num_tens >= 3).astype(int)

    logs["FANTASY_PTS"] = (
        0.5  * logs["FGM"]
        - 0.5  * logs["FGMI"]
        + 0.25 * logs["FTM"]
        - 0.5  * logs["FTMI"]
        + 0.5  * logs["FG3M"]
        + 1.2  * logs["REB"]
        + 1.5  * logs["AST"]
        + 3    * logs["STL"]
        + 3    * logs["BLK"]
        - 1    * logs["TOV"]
        + 2    * logs["DD"]
        + 5    * logs["TD"]
        + 1    * logs["PTS"]
    )
    return logs

def add_features(logs):
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs = logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    g = logs.groupby("PLAYER_ID")

    # Rolling averages of past games only (shift(1) keeps the current game out)
    for stat in ["FANTASY_PTS", "MIN", "PTS", "REB", "AST", "FGA", "TOV"]:
        logs[f"{stat}_avg5"] = g[stat].transform(lambda s: s.shift(1).rolling(5).mean())
        logs[f"{stat}_avg15"] = g[stat].transform(lambda s: s.shift(1).rolling(15).mean())

    # Home or away (away games have "@" in the matchup text)
    logs["IS_HOME"] = (~logs["MATCHUP"].str.contains("@")).astype(int)

    # Days of rest since the player's last game
    logs["REST_DAYS"] = g["GAME_DATE"].diff().dt.days.clip(upper=7)

    return logs

if __name__ == "__main__":
    logs = load_logs()
    logs = add_fantasy_points(logs)
    logs = add_features(logs)

    print(logs.shape)
    print(logs[["PLAYER_NAME", "GAME_DATE", "FANTASY_PTS"]].head())