import sys
import os
import gzip
import pickle
import numpy as np
import torch
from keras.models import load_model
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import optuna

solvation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(solvation_path)

import smi
import mdm
import gnn

DATA_PATH = '../data/'
BS = gnn.config.bs
N_SPLITS = 10  # Number of folds for cross-validation

def load_pickled_data(file_path):
    with gzip.open(file_path, "rb") as f:
        return pickle.load(f)

def setup_data_loader(data):
    loader = DataLoader(data, batch_size=BS, shuffle=False, drop_last=False)
    return loader

def setup_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = gnn.gnn_model.GNN(n_features=gnn.config.n_features).to(device)
    gnn_model.load_state_dict(torch.load(gnn.config.best_model))
    smi_model = load_model(smi.config.best_model)
    mdm_model = load_model(mdm.config.best_model)
    return gnn_model, smi_model, mdm_model, device

def objective(trial, val_data, smi_x_val, mdm_x_val, y_val, gnn_model, smi_model, mdm_model, device):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, val_index in kf.split(val_data):
        # Creating DataLoaders for the validation split
        current_val_data = [val_data[i] for i in val_index]
        current_val_loader = setup_data_loader(current_val_data)

        # Weights to optimize
        weights = {
            'gnn': trial.suggest_float("weight_gnn", 0, 1),
            'mdm': trial.suggest_float("weight_mdm", 0, 1),
            'smi': trial.suggest_float("weight_smi", 0, 1)
        }

        # Normalize weights
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total

        # Predict probabilities
        _, proba_gnn = gnn.gnn_utils.test_fn_plotting(current_val_loader, gnn_model, device)
        proba_mdm = mdm_model.predict(mdm_x_val[val_index]).ravel()
        proba_smi = smi_model.predict(smi_x_val[val_index]).ravel()

        # Weighted average of probabilities
        ensemble_proba = weights['gnn'] * proba_gnn + weights['mdm'] * proba_mdm + weights['smi'] * proba_smi

        # Compute mean squared error
        mse = mean_squared_error(y_val[val_index], ensemble_proba)
        mse_scores.append(mse)

    # Return average MSE across all folds
    return np.mean(mse_scores)


def main():
    val_data = load_pickled_data(os.path.join(DATA_PATH, "val.pkl.gz"))
    smi_x_val = np.loadtxt("../smi/smi_input/x_val.txt")
    mdm_x_val = np.loadtxt("../mdm/input/x_val.txt")
    y_val = np.loadtxt("../input/y_val.txt")

    gnn_model, smi_model, mdm_model, device = setup_models()

    # Setup and run the Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, val_data, smi_x_val, mdm_x_val, y_val, gnn_model, smi_model, mdm_model, device), n_trials=50)

    # Best weights found
    best_weights = study.best_params
    print("Best weights:", best_weights)

    # Save the results to a text file
    with open("optimization_results.txt", "w") as f:
        f.write(f"Best weights: {best_weights}\n")

if __name__ == '__main__':
    main()
