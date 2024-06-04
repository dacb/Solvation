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
    val_X = load_pickled_data(os.path.join(data_path, "val.pkl.gz"))
    test_X = load_pickled_data(os.path.join(data_path, "test.pkl.gz"))
    
    val_loader = DataLoader(val_X, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_X, batch_size=batch_size, shuffle=False, drop_last=False)
    return val_loader, test_loader

# Load models and make predictions
def load_models_and_predict(val_loader, test_loader, device):
    gnn_model = gnn.gnn_model.GNN(n_features=gnn.config.n_features).to(device)
    gnn_model.load_state_dict(torch.load(gnn.config.best_model))
    _, gnn_val_pred = gnn.gnn_utils.test_fn_plotting(val_loader, gnn_model, device)
    _, gnn_test_pred = gnn.gnn_utils.test_fn_plotting(test_loader, gnn_model, device)

    smi_x_val = np.loadtxt("./data/x_val.txt")
    smi_x_test = np.loadtxt("./data/x_test.txt")
    smi_model = load_model(smi.config.best_model)
    smi_val_pred = smi_model.predict(smi_x_val).ravel()
    smi_test_pred = smi_model.predict(smi_x_test).ravel()

    mdm_x_val = np.loadtxt("./data/x_val.txt")
    mdm_x_test = np.loadtxt("./data/x_test.txt")
    mdm_model = load_model(mdm.config.best_model)
    mdm_val_pred = mdm_model.predict(mdm_x_val).reshape(-1,)
    mdm_test_pred = mdm_model.predict(mdm_x_test).reshape(-1,)

    return gnn_val_pred, smi_val_pred, mdm_val_pred, gnn_test_pred, smi_test_pred, mdm_test_pred

# Calculate performance metrics
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, r2, mae

# Find closest rows based on Euclidean distance
def find_closest_rows(test_data, valid_data, columns_of_interest):
    closest_indices = []
    for idx, test_row in test_data.iterrows():
        distances = valid_data[columns_of_interest].apply(lambda row: distance.euclidean(row, test_row[columns_of_interest]), axis=1)
        closest_indices.append(distances.nsmallest(5).index.tolist())
    return closest_indices

# Main execution function
def main():
    data_path = './data/'
    batch_size = gnn.config.bs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_loader, test_loader = setup_data_loaders(data_path, batch_size)
    gnn_val_pred, smi_val_pred, mdm_val_pred, gnn_test_pred, smi_test_pred, mdm_test_pred = load_models_and_predict(val_loader, test_loader, device)

    valid_data = pd.read_csv('./data/val.csv')
    test_data = pd.read_csv("./data/test.csv")
    y_true = valid_data['log_sol']

    valid_data['gnn_predict'] = pd.Series(gnn_val_pred)
    valid_data['mdm_predict'] = pd.Series(mdm_val_pred)
    valid_data['smi_predict'] = pd.Series(smi_val_pred)
    valid_data['gnn_mse'] = (valid_data['gnn_predict'] - y_true) ** 2
    valid_data['mdm_mse'] = (valid_data['mdm_predict'] - y_true) ** 2
    valid_data['smi_mse'] = (valid_data['smi_predict'] - y_true) ** 2

    columns_of_interest = ['MW', 'vol', 'nAromAtom', 'nAromBond', 'nHeavyAtom', 'nBridgehead', 'nHBDon', 'nHBAcc', 'nRot']
    closest_rows_indices = find_closest_rows(test_data, valid_data, columns_of_interest)

    best_model = []
    for indices in closest_rows_indices:
        avg_gnn_mse = valid_data.loc[indices, 'gnn_mse'].mean()
        avg_mdm_mse = valid_data.loc[indices, 'mdm_mse'].mean()
        avg_smi_mse = valid_data.loc[indices, 'smi_mse'].mean()
        min_mse = min(avg_gnn_mse, avg_mdm_mse, avg_smi_mse)
        if min_mse == avg_gnn_mse:
            best_model.append('gnn')
        elif min_mse == avg_mdm_mse:
            best_model.append('mdm')
        else:
            best_model.append('smi')

    test_data['best_model'] = best_model
    model_counts = test_data['best_model'].value_counts()

    final_predictions = []
    for idx, row in test_data.iterrows():
        if row['best_model'] == 'gnn':
            final_predictions.append(gnn_test_pred[idx])
        elif row['best_model'] == 'mdm':
            final_predictions.append(mdm_test_pred[idx])
        else:
            final_predictions.append(smi_test_pred[idx])

    test_data['final_prediction'] = final_predictions

    y_test_true = test_data['log_sol'] # Assuming test data also has 'log_sol' for true values
    rmse, r2, mae = calculate_metrics(y_test_true, test_data['final_prediction'])
    print(f"Final Model - RMSE: {rmse}, R2: {r2}, MAE: {mae}")

    print("Model Counts in Best Model Selection:")
    print(model_counts)

if __name__ == "__main__":
    main()
