"""Predict yield using a trusted local baseline checkpoint and label-free CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.data.loader import EXPECTED_FEATURES, ID_COL


def feature_frame(frame):
    missing = set(EXPECTED_FEATURES)-set(frame.columns)
    extra = set(frame.columns)-set(EXPECTED_FEATURES)-{ID_COL}
    if missing or extra:
        raise ValueError(f"Input schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    if not frame.columns.is_unique or len(frame)==0:
        raise ValueError("Input must contain rows and unique column names")
    features = frame[EXPECTED_FEATURES].astype(float)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Input features contain NaN or infinite values")
    return features


def predict(model_path, input_path, output_path):
    model_path, input_path, output_path = map(Path,[model_path,input_path,output_path])
    if output_path.resolve() in {model_path.resolve(),input_path.resolve()}:
        raise ValueError("Output must not overwrite the input or model")
    frame = pd.read_csv(input_path)
    features = feature_frame(frame)
    # joblib/pickle must only be loaded from sources the caller trusts.
    model = joblib.load(model_path)
    expected = list(getattr(model,"feature_names_in_",[]))
    if expected != EXPECTED_FEATURES:
        raise ValueError("Checkpoint feature schema mismatch")
    values = model.predict(features)
    if not np.isfinite(values).all(): raise ValueError("Non-finite predictions")
    result = frame[[ID_COL]].copy() if ID_COL in frame.columns else pd.DataFrame(index=frame.index)
    result["predicted_yield"] = values
    output_path.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output_path,index=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True,type=Path)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    print(predict(args.model,args.input,args.output).to_string(index=False))


if __name__=="__main__": main()
