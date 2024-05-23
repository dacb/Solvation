import sys
import os
import gzip
import pickle
import numpy as np
import torch
from keras.models import load_model
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_squared_error
import optuna
import smi
import mdm
import gnn

DATA_PATH = '/home/fostooq/solubility-prediction-paper/data/'
BS = gnn.config.bs

def load_pickled_data(file_path):
    with gzip.open(file_path, "rb") as f:
        return pickle.load(f)

def setup_data_loaders():
    val_X = load_pickled_data(os.path.join(DATA_PATH, "val.pkl.gz"))
    test_X = load_pickled_data(os.path.join(DATA_PATH, "test.pkl.gz"))
    val_loader = DataLoader(val_X, batch_size=BS, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_X, batch_size=BS, shuffle=False, drop_last=False)
    return val_loader, test_loader

def setup_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = gnn.gnn_model.GNN(n_features=gnn.config.n_features).to(device)
    gnn_model.load_state_dict(torch.load(gnn.config.best_model))
    smi_model = load_model(smi.config.best_model)
    mdm_model = load_model(mdm.config.best_model)
    return gnn_model, smi_model, mdm_model, device


def objective(trial, val_loader, gnn_model, smi_model, mdm_model, smi_x_val, mdm_x_val, y_val, device):
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
    _, proba_gnn = gnn.gnn_utils.test_fn_plotting(val_loader, gnn_model, device)
    proba_mdm = mdm_model.predict(mdm_x_val).ravel()
    proba_smi = smi_model.predict(smi_x_val).ravel()

    # Weighted average of probabilities
    ensemble_proba = [weights['gnn'] * g + weights['mdm'] * m + weights['smi'] * s for g, m, s in zip(proba_gnn, proba_mdm, proba_smi)]

    # Compute mean squared error
    mse = mean_squared_error(y_val, ensemble_proba)
    return mse

def main():
    smi_x_val = np.loadtxt("smi_input/x_val.txt")
    mdm_x_val = np.loadtxt("input/x_val.txt")
    y_val = np.loadtxt("input/y_val.txt")

    val_loader, test_loader = setup_data_loaders()
    gnn_model, smi_model, mdm_model, device = setup_models()

    # Setup and run the Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, val_loader, gnn_model, smi_model, mdm_model, smi_x_val, mdm_x_val, y_val, device), n_trials=100)
    
    # Best weights found
    best_weights = study.best_params
    print("Best weights:", best_weights)

     # Save the results to a text file
    with open("optimization_results.txt", "w") as f:
        f.write(f"Best weights: {best_weights}\n")

if __name__ == '__main__':
    main()
