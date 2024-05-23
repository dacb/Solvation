<https://img.shields.io/github/license/FaizaAB/Solvation>

# Solvation Meta Predictor



Code for the ML models are optained from: Predicting Aqueous Solubility of Organic Molecules Using Deep Learning Models with Varied Molecular Representations (https://pubs.acs.org/doi/full/10.1021/acsomega.2c00642)

#### Usage
1. Download data from https://figshare.com/s/542fb80e65742746603c and save it as data.csv in the ./data folder
2. Generate Pybel coordinates and MDM features by running create_data.py at the ./data folder
3. To train MDM, GNN, and SMI models run train.py at ./mdm, ./gnn, and ./smi folderes respectively.
4. To make predictions use predict.ipynb files at each model folders.
5. To ansemble the models run CV.py , Optuna.py and KNN.py.
6. To compare predictions from individual models with ensemble methods use ensemble_prediction.ipynb
