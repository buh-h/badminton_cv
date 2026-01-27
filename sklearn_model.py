from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate
import pandas as pd
import numpy as np
import utils.general 

# FEATURE_COLS = ["a", "b", "c", "dy", "start_y", "length", "peak", "peak_frame", "speed"]
FEATURE_COLS = ["dy", "start_y", "length", "peak", "peak_frame", "speed"]
N_SPLITS = 5

def prepare_data(features):
    mapping = {"Kill": "Smash", "Push": "Drive", "Block": "Drop"}
    features['shot'] = features['shot'].replace(mapping)
    
    print(features)
    X = features[FEATURE_COLS].to_numpy(dtype=np.float32)
    groups = features["match_id"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(features["shot"])

    return X, y, groups

if __name__ == "__main__":
    annotations_file = "dataset/shots.json"
    rally_dir = "dataset/rallies"
    shots_dict, rallies_df = utils.general.get_dataset(annotations_file, rally_dir)
    all_features = utils.general.get_features(shots_dict, rallies_df)

    X, y, groups = prepare_data(all_features)

    print("Building model...")
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("Running grouped cross-validation...")
    cv = GroupKFold(n_splits=N_SPLITS)
    scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring={
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro",
        },
        return_train_score=False
    )

    print("\nCross-validation results:")
    for metric in ["test_precision", "test_recall", "test_f1"]:
        values = scores[metric]
        print(f"{metric}: {values.mean():.3f} ± {values.std():.3f}")


    model.fit(X, y)
    importances = model.feature_importances_

    print("Feature importances:")
    for f, imp in sorted(zip(FEATURE_COLS, importances),
                          key=lambda x: x[1], reverse=True):
        print(f"{f:>12}: {imp:.4f}")