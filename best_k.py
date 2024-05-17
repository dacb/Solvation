import os
import numpy as np
import gzip
import pickle
from keras.models import load_model
from torch_geometric.data import DataLoader
import torch
import smi
import mdm
import gnn
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

# Function to load pickled data
def load_pickled_data(file_path):
    with gzip.open(file_path, "rb") as f:
        return pickle.load(f)

# Setup data loaders
def setup_data_loaders(data_path, batch_size):
    train_X = load_pickled_data(os.path.join(data_path, "train.pkl.gz"))
    val_X = load_pickled_data(os.path.join(data_path, "val.pkl.gz"))
    test_X = load_pickled_data(os.path.join(data_path, "test.pkl.gz"))

    val_loader = DataLoader(val_X, batch_size=batch_size, shuffle=False, drop_last=False)
    return val_loader

# Load models and make predictions
def load_models_and_predict(val_loader, device):
    gnn_model = gnn.gnn_model.GNN(n_features=gnn.config.n_features).to(device)
    gnn_model.load_state_dict(torch.load(gnn.config.best_model))
    _, gnn_pred = gnn.gnn_utils.test_fn_plotting(val_loader, gnn_model, device)

    smi_x_val = np.loadtxt("./smi_input/x_val.txt")
    smi_model = load_model(smi.config.best_model)
    smi_pred = smi_model.predict(smi_x_val).ravel()

    mdm_x_val = np.loadtxt("./input/x_val.txt")
    mdm_model = load_model(mdm.config.best_model)
    mdm_pred = mdm_model.predict(mdm_x_val).reshape(-1,)

    return gnn_pred, smi_pred, mdm_pred

# Calculate performance metrics
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    return rmse, r2, mae, spearman_corr

# Main execution function
def main():
    data_path = './data/'
    batch_size = gnn.config.bs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_loader = setup_data_loaders(data_path, batch_size)
    gnn_pred, smi_pred, mdm_pred = load_models_and_predict(val_loader, device)

    test_data = pd.read_csv("./New_test_data.csv")
    valid_data = pd.read_csv('./data/val.csv')
    y_true = valid_data['log_sol']

    valid_data['gnn_predict'] = pd.Series(gnn_pred)
    valid_data['mdm_predict'] = pd.Series(mdm_pred)
    valid_data['smi_predict'] = pd.Series(smi_pred)
    valid_data['gnn_error'] = (valid_data['gnn_predict'] - y_true).abs()
    valid_data['mdm_error'] = (valid_data['mdm_predict'] - y_true).abs()
    valid_data['smi_error'] = (valid_data['smi_predict'] - y_true).abs()

    # Calculate and print metrics for each model prediction
    rmse, r2, mae, spearman_corr = calculate_metrics(y_true, valid_data['gnn_predict'])
    print(f"GNN Model - RMSE: {rmse}, R2: {r2}, MAE: {mae}, Spearman: {spearman_corr}")

    rmse, r2, mae, spearman_corr = calculate_metrics(y_true, valid_data['mdm_predict'])
    print(f"MDM Model - RMSE: {rmse}, R2: {r2}, MAE: {mae}, Spearman: {spearman_corr}")

    rmse, r2, mae, spearman_corr = calculate_metrics(y_true, valid_data['smi_predict'])
    print(f"SMI Model - RMSE: {rmse}, R2: {r2}, MAE: {mae}, Spearman: {spearman_corr}")

if __name__ == "__main__":
    main()
