import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
import re
import json

def get_dataset(shots_path="dataset/shots.json", rally_dir="dataset/rallies"):
    print(f"Pulling shot data from {shots_path}, and bird positional data from {rally_dir}")
    with open(shots_path) as f:
        shots_dict = json.load(f)

    rally_pattern = re.compile(r"rally-seg(\d+)", re.IGNORECASE)

    indv_dfs = []
    for match_dir in Path(rally_dir).iterdir():
        if not match_dir.is_dir():
            continue

        match_id = match_dir.name

        for csv_path in match_dir.glob("*.csv"):
            stem = csv_path.stem
            match = rally_pattern.match(stem)
            rally_id = int(match.group(1)) if match else None

            df = pd.read_csv(csv_path)
            df["match_id"] = match_id
            df["rally_id"] = rally_id

            indv_dfs.append(df)

    rallies_df = pd.concat(indv_dfs, ignore_index=True)
    rallies_df = rallies_df.sort_values(
        ["match_id", "rally_id", "Frame"]
    ).reset_index(drop=True)

    return shots_dict, rallies_df

def remove_dup(shot_df):
    filtered_rows = []
    shifted_frames = 0
    prev_x, prev_y = None, None

    for _, row in shot_df.iterrows():
        # Also removes non-visible rows, but keeps the gap in frames
        if prev_x is not None and row["X"] == prev_x and row["Y"] == prev_y:
            continue
        
        new_row = row.copy()
        if row["Visibility"] == 0:
            new_row["Y"] = None
        new_row["Frame"] = shifted_frames
        filtered_rows.append(new_row)
        
        prev_x, prev_y = row["X"], row["Y"]
        shifted_frames += 1

    return pd.DataFrame(filtered_rows)

def smooth_y(shot_df):
    shot_df['Y'] = savgol_filter(shot_df['Y'], window_length=5, polyorder=2)
    return

def fit_to_curve(shot_dict, rally_df, end_frame):
    shot_df = rally_df[
        (rally_df["Frame"] >= shot_dict["frame_number"]) & 
        (rally_df["Frame"] <= end_frame)
    ].copy()

    shot_df = remove_dup(shot_df)
    if len(shot_df) < 5:
        return None
    # shot_df["Y"] = shot_df["Y"].interpolate(method='quadratic')
    shot_df = shot_df.dropna(subset=["Y"])
    shot_df['Y'] = savgol_filter(shot_df['Y'], window_length=5, polyorder=2)

    x = shot_df["Frame"].to_numpy()
    y = shot_df["Y"].to_numpy()

    # a = 0.2
    # y_transformed = y - (a * x**2)
    # b, c = np.polyfit(x, y_transformed, 1)
    
    a, b, c = np.polyfit(x, y, 2)

    return {
        "shot": shot_dict["shot"],
        "a": a,
        "b": b,
        "c": c,
        "last_frame": shot_df["Frame"].iloc[-1],
        "match_id": shot_df["match_id"].iloc[0],
        "rally_id": shot_df["rally_id"].iloc[0]
    }

def get_physical_info(shot_dict, rally_df, end_frame):
    shot_df = rally_df[
        (rally_df["Frame"] >= shot_dict["frame_number"]) & 
        (rally_df["Frame"] <= end_frame)
    ].copy()

    shot_df = remove_dup(shot_df)
    if len(shot_df) < 5:
        return None
    
    peak = shot_df["Y"].min()
    shot_df["Y"] = shot_df["Y"].interpolate(method='quadratic')
    shot_df['Y'] = savgol_filter(shot_df['Y'], window_length=5, polyorder=2)
    shot_df = shot_df.dropna(subset=["Y"])
    
    shot_df["speed_y"] = shot_df["Y"].diff().abs()
    
    return {
        "shot": shot_dict["shot"],
        "start_y": shot_df["Y"].iloc[0],
        "dy": shot_df["Y"].iloc[-1] - shot_df["Y"].iloc[0],
        "speed": shot_df["speed_y"].mean(),
        "peak": peak,
        "peak_frame": shot_df.loc[shot_df["Y"].idxmin(), "Frame"],
        "length": shot_df["Frame"].iloc[-1],
        "match_id": shot_df["match_id"].iloc[0],
        "rally_id": shot_df["rally_id"].iloc[0]
    }




def get_features(shots_dict, rallies_df):
    all_shots = []
    grouped_rallies = rallies_df.groupby(["match_id", "rally_id"])
    for match_id, rallies in shots_dict.items():
        for rally_num, shots in rallies.items():
            try:
                rally_df = grouped_rallies.get_group((match_id, int(rally_num)))
            except KeyError:
                continue

            prev_shot = None
            for shot in shots: 
                if not prev_shot or (prev_shot["shot"] == "Serve"):
                    prev_shot = shot
                    continue
                
                curve_features = fit_to_curve(prev_shot, rally_df, shot["frame_number"])
                physical_features = get_physical_info(prev_shot, rally_df, shot["frame_number"])
                if curve_features and physical_features:
                    features = curve_features | physical_features
                elif curve_features:
                    features = curve_features
                else:
                    features = physical_features
                if features: 
                    all_shots.append(features)
                prev_shot = shot

    return pd.DataFrame(all_shots)



            





