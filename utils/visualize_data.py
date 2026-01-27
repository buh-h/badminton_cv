import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import argparse
import json
import general as general
import os

def feature_hist(shots_dict, rallies_df, feature_cols, out_dir: str):
    all_features = general.get_features(shots_dict, rallies_df)
    shot_types = ["Clear", "Drop", "Smash", "Lift", "Net", "Kill", "Drive", "Push", "Block"]
    os.makedirs(out_dir, exist_ok=True)

    for shot in shot_types: 
        filtered_features = all_features[all_features["shot"] == shot]
        for f in feature_cols:
            fig, ax = plt.subplots(figsize=(12,4))
            ax.hist(filtered_features[f], bins='auto', color='#3498db', edgecolor='black', alpha=0.7)
            ax.set_title(f"{f} for {shot}")
            ax.set_xlabel(f)
            ax.set_ylabel("Count")
            ax.legend()
            fig.savefig(f"{out_dir}/{f}_{shot}_hist.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

def feature_box(shots_dict, rallies_df, feature_cols, out_dir: str):
    all_features = general.get_features(shots_dict, rallies_df)
    shot_types = ["Clear", "Drop", "Smash", "Lift", "Net", "Kill", "Drive", "Push", "Block"]
    os.makedirs(out_dir, exist_ok=True)
    print(all_features)

    # for shot in shot_types: 
    # filtered_features = all_features[all_features["shot"] == shot]
    for f in feature_cols:
        data_to_plot = [
            all_features.loc[all_features["shot"] == s, f].dropna().values
            for s in shot_types
        ]
        fig, ax = plt.subplots(figsize=(6,8))
        # ax.boxplot(filtered_features[f])
        ax.boxplot(data_to_plot, labels=shot_types, patch_artist=True, notch=False,
            medianprops={'color': 'black', 'linewidth': 2},
            boxprops={'facecolor': '#3498db', 'edgecolor': 'black'},
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5},
            showfliers=False
        )
        ax.set_title(f'Comparison of {f} across Shot Types', fontsize=16)
        ax.set_ylabel('Coefficient Value')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.savefig(f"{out_dir}/{f}_comp_box.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

def visualize_curves(shots_dict, rallies_df, match: str, rally_num: int, out_dir: str):
    # hits_dict = shots_dict[f"{match}_seg{rally_num}"]
    df = rallies_df[
        (rallies_df["match_id"] == match) & 
        (rallies_df["rally_id"] == rally_num)
    ]
    curves_df = general.get_features(shots_dict, rallies_df)
    filtered_curves = curves_df[curves_df["shot"] == "Lift"]
    print(filtered_curves)

    fig, ax = plt.subplots(figsize=(12,4))

    for _, row in filtered_curves.iterrows():
        # x values for plotting the quadratic
        x_fit = np.arange(0, row["last_frame"] + 1)
        y_fit = row["a"]*x_fit**2 + row["b"]*x_fit + row["c"]
        
        ax.plot(x_fit, y_fit)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Y")
    ax.set_title(f"{match}_seg{rally_num}")
    ax.legend()
    ax.grid(True)
    fig.savefig(f"{out_dir}/_curves{rally_num}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    
def visualize_y(shots_dict, rallies_df, match: str, rally_num: int, out_dir: str):
    print(f"Creating visual for rally {rally_num} in {match}")
    hits_list = shots_dict[match][str(rally_num)]
    df = rallies_df[
        (rallies_df["match_id"] == match) & 
        (rallies_df["rally_id"] == rally_num)
    ]
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax.scatter(df['Frame'], df['Y'], color='blue', label="Shuttle")

    for hit in hits_list:
        ax.axvline(hit["frame_number"], linestyle="--", alpha=0.7, color="red")

        ax.text(
            hit["frame_number"], 
            ax.get_ylim()[1] + 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
            hit["shot"],
            rotation=90,
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=8,
            color="red"
        )
    

    ax.set_xlabel("Frame")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)

    dir_path = os.path.join(out_dir, match)
    os.makedirs(dir_path, exist_ok=True)
    output_file = os.path.join(dir_path, f"plot{rally_num}.png")

    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rallies', type=str, default="../dataset/rallies", help='Directory containing shuttle position csvs')
    parser.add_argument('-s', '--shots', type=str, default="../dataset/shots.json", help='JSON file containing shot labels')
    parser.add_argument('-o', '--output', type=str, default="../visualizations", help='Directory to ouput graphs')
    parser.add_argument('-m', '--match', type=str, default=None, help='Match name being visualized')
    parser.add_argument('-n', '--num', type=int, default=None, help='Rally number of the match being visualized')
    args = parser.parse_args()

    shots_dict, rallies_df = general.get_dataset(args.shots, args.rallies)
    match_name = args.match
    seg = args.num
    output = args.output
    """
    {
        "shot": shot_dict["shot"],
        "start_y": shot_df["Y"].iloc[0],
        "max_speed": shot_df["speed_y"].max(),
        "peak": shot_df["Y"].min(),
        "peak_frame": shot_df["Y"].id,
        "length": shot_df["Frame"].iloc[-1],
        "match_id": shot_df["match_id"].iloc[0],
        "rally_id": shot_df["rally_id"].iloc[0]
    }
    """
    features = ["a", "b", "c", "start_y", "dy", "speed", "peak", "peak_frame", "length"]
    # features = ["a", "b", "c"]
    feature_box(shots_dict, rallies_df, feature_cols=features, out_dir=args.output)
    # if match_name and seg:
    #     visualize_y(shots_dict, rallies_df, match=match_name, rally_num=int(seg), out_dir=args.output)
    #     visualize_curves(shots_dict, rallies_df, match=match_name, rally_num=int(seg), out_dir=args.output)
    # elif match_name:
    #     for rally in shots_dict[match_name].keys():
    #         visualize_y(shots_dict, rallies_df, match=match_name, rally_num=int(rally), out_dir=args.output)
    #         visualize_curves(shots_dict, rallies_df, match=match_name, rally_num=int(rally), out_dir=args.output)
            
    # elif seg:

    # else:

    