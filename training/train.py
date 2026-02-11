import os, json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_X = os.getenv("DATA_X", os.path.join(BASE_DIR, "data", "engie_X.csv"))
DATA_Y = os.getenv("DATA_Y", os.path.join(BASE_DIR, "data", "engie_Y.csv"))
OUTDIR = os.getenv("OUTDIR", os.path.join(BASE_DIR, "..", "inference", "models"))
os.makedirs(OUTDIR, exist_ok=True)

print("Paths")
print("  DATA_X :", DATA_X)
print("  DATA_Y :", DATA_Y)
print("  OUTDIR :", OUTDIR)

# Load + merge
print("\n[Load] Reading CSVs...")
X = pd.read_csv(DATA_X, sep=";")
Y = pd.read_csv(DATA_Y, sep=";")
df = X.merge(Y, on="ID").set_index("ID")

print("\nShapes")
print("  X shape:", X.shape)
print("  Y shape:", Y.shape)
print("  Merged shape:", df.shape)

# Keep only rows where TARGET is non-negative
n0 = len(df)
df = df[df["TARGET"] >= 0].copy()
print("\n[Filter] TARGET >= 0")
print(f"  Rows before: {n0:,} | after: {len(df):,} | removed: {n0 - len(df):,}")

print("\n[Data] Using ALL MAC_CODE values")
print(df["MAC_CODE"].value_counts().head(10).to_string())

# Feature pruning
FEATURES_TO_DELETE = [
    "Rotor_speed_max", "Rotor_speed_min",
    "Generator_speed_max", "Generator_speed_min",
    "Generator_converter_speed_max", "Generator_converter_speed_min",
    "Pitch_angle_min", "Pitch_angle_max",
    "Nacelle_angle_min", "Nacelle_angle_max",
    "Grid_frequency_min", "Grid_frequency_max",
    "Grid_voltage_min", "Grid_voltage_max",
    "Generator_converter_speed",

    "Gearbox_bearing_1_temperature_max", "Gearbox_bearing_1_temperature_min",
    "Gearbox_bearing_2_temperature_max", "Gearbox_bearing_2_temperature_min",
    "Gearbox_bearing_2_temperature",
    "Gearbox_inlet_temperature_max", "Gearbox_inlet_temperature_min",
    "Gearbox_oil_sump_temperature_max", "Gearbox_oil_sump_temperature_min",
    "Generator_bearing_1_temperature_max", "Generator_bearing_1_temperature_min",
    "Generator_bearing_2_temperature_max", "Generator_bearing_2_temperature_min",
    "Generator_bearing_2_temperature",
    "Generator_stator_temperature_max", "Generator_stator_temperature_min",
    "Hub_temperature_max", "Hub_temperature_min",
    "Nacelle_temperature_max", "Nacelle_temperature_min",
    "Rotor_bearing_temperature_max", "Rotor_bearing_temperature_min",
    "Outdoor_temperature_max", "Outdoor_temperature_min",

    "Absolute_wind_direction_c",
    "Nacelle_angle_c",

    "Rotor_speed", "Rotor_speed_std",
    "Rotor_bearing_temperature", "Rotor_bearing_temperature_std",
]

def drop_features(df, drop_list, verbose=True):
    before_cols = df.columns.tolist()
    before = df.shape[1]
    df2 = df.drop(columns=drop_list, errors="ignore")
    after = df2.shape[1]
    if verbose:
        matched = [c for c in drop_list if c in before_cols]
        print(f"\n[Feature pruning] Original: {before} cols | Final: {after} cols | Deleted: {before-after}")
        print(f"  Matched drop columns: {len(matched)} / {len(drop_list)}")
    return df2

df = drop_features(df, FEATURES_TO_DELETE, verbose=True)

# Encode turbine (back)
print("\n[Encode] Label encode MAC_CODE -> MAC_CODEencoded")
le = LabelEncoder()
df["MAC_CODEencoded"] = le.fit_transform(df["MAC_CODE"])
print("  Classes:", list(le.classes_))

# Sort BEFORE ffill to ensure proper temporal order per turbine
print("\n[Sort] Sort by MAC_CODE, then Date_time (before ffill)")
df = df.sort_values(["MAC_CODE", "Date_time"]).reset_index(drop=False)

# Feature columns
exclude = ["ID", "MAC_CODE", "TARGET", "Date_time"]
feature_cols = [c for c in df.columns if c not in exclude]

print("\nFeature set")
print("  Excluded:", exclude)
print("  n_features:", len(feature_cols))

# Impute per turbine: ffill then median fallback
print("\n[Impute] ffill per turbine then median fallback")
na_before = int(df[feature_cols].isna().sum().sum())
print("  Missing cells before:", f"{na_before:,}")

df[feature_cols] = df.groupby("MAC_CODE")[feature_cols].ffill()
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

na_after = int(df[feature_cols].isna().sum().sum())
print("  Missing cells after :", f"{na_after:,}")

# Temporal split globally by Date_time
df_sorted = df.sort_values("Date_time").reset_index(drop=True)
Xfull = df_sorted[feature_cols].values
yfull = df_sorted["TARGET"].values

n = len(df_sorted)
tr_end, val_end = int(0.7 * n), int(0.85 * n)
Xtrain, Xval, Xtest = Xfull[:tr_end], Xfull[tr_end:val_end], Xfull[val_end:]
ytrain, yval, ytest = yfull[:tr_end], yfull[tr_end:val_end], yfull[val_end:]

print("\nSplit sizes")
print(f"  n_total: {n:,}")
print(f"  train: {len(Xtrain):,} | val: {len(Xval):,} | test: {len(Xtest):,}")
print("  Xtrain shape:", Xtrain.shape, "ytrain shape:", ytrain.shape)

# Scale
scalerX = StandardScaler()
Xtrain_s = scalerX.fit_transform(Xtrain)
Xval_s = scalerX.transform(Xval)
Xtest_s = scalerX.transform(Xtest)

scalery = StandardScaler()
ytrain_s = scalery.fit_transform(ytrain.reshape(-1, 1)).ravel()
yval_s = scalery.transform(yval.reshape(-1, 1)).ravel()
ytest_s = scalery.transform(ytest.reshape(-1, 1)).ravel()

print("\n[Scale] Done")

# Model
optimizer = tf.keras.optimizers.Adam(learning_rate=1.66e-4)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(Xtrain_s.shape[1],)),
    tf.keras.layers.Dropout(0.0),
    tf.keras.layers.Dense(224, activation="relu"),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer=optimizer, loss="mae")

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

print("\n[Train] Start fit...")
model.fit(
    Xtrain_s, ytrain_s,
    validation_data=(Xval_s, yval_s),
    epochs=200,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)

# Save artifacts (back including label encoder)
print("\n[Save] Writing artifacts...")
model.save(os.path.join(OUTDIR, "model.keras"))
joblib.dump(scalerX, os.path.join(OUTDIR, "scaler_x.pkl"))
joblib.dump(scalery, os.path.join(OUTDIR, "scaler_y.pkl"))
joblib.dump(le, os.path.join(OUTDIR, "label_encoder.pkl"))
json.dump(feature_cols, open(os.path.join(OUTDIR, "feature_cols.json"), "w"))
json.dump(
    {"n_features": len(feature_cols), "uses_mac_code_encoded": True},
    open(os.path.join(OUTDIR, "metadata.json"), "w")
)

print("Training done. Artifacts saved to", OUTDIR)
