import os
import time
import pandas as pd
from nba_api.stats.endpoints import playergamelogs

CSV_PATH = "game_logs.csv"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
SEASON_TYPES = ["Regular Season", "Playoffs"]


def load_logs():
    # Reuse the saved file so you don't re-download on every run
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)

    all_logs = []
    for season in SEASONS:
        for season_type in SEASON_TYPES:
            print(f"Pulling {season} {season_type}...")
            df = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable=season_type,
            ).get_data_frames()[0]
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
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

    # Playoff flag
    logs["IS_PLAYOFFS"] = (logs["SEASON_TYPE"] == "Playoffs").astype(int)

    return logs


if __name__ == "__main__":
    logs = load_logs()
    print(logs["SEASON_TYPE"].value_counts())

    logs = add_fantasy_points(logs)
    logs = add_features(logs)

    feature_cols = [c for c in logs.columns if c.endswith(("_avg5", "_avg15"))] + [
        "IS_HOME", "REST_DAYS", "IS_PLAYOFFS"
    ]

    print("Before:", logs.shape)
    data = logs.dropna(subset=feature_cols)
    print("After:", data.shape)

    # Train on the first three seasons, test on 2025-26 (regular season + playoffs)
    train = data[data["GAME_DATE"] < "2025-09-01"]
    test = data[data["GAME_DATE"] >= "2025-09-01"]

    X_train, y_train = train[feature_cols], train["FANTASY_PTS"]
    X_test, y_test = test[feature_cols], test["FANTASY_PTS"]

    print("Train:", train.shape, "Test:", test.shape)
    print("Latest train game:", train["GAME_DATE"].max())
    print("Earliest test game:", test["GAME_DATE"].min())