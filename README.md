<img alt="GitHub License" src="https://img.shields.io/github/license/FaizaAB/Solvation">

# Solvation Meta Predictor



Code for the ML models are optained from: Predicting Aqueous Solubility of Organic Molecules Using Deep Learning Models with Varied Molecular Representations (https://pubs.acs.org/doi/full/10.1021/acsomega.2c00642)

#### Usage
1. Download data from https://figshare.com/s/542fb80e65742746603c and save it as `data.csv` in the `./data` folder
2. Generate Pybel coordinates and MDM features by running create_data.py at the `./data` folder
3. To train MDM, GNN, and SMI models run train.py at `./mdm`, `./gnn`, and `./smi` folderes respectively.
4. To make predictions use `predict.ipynb` files at each model folders.
5. To ansemble the models run `CV.py`, `Optuna.py` and `KNN.py`.
6. To compare predictions from individual models with ensemble methods use `ensemble_prediction.ipynb`

Sure, I can help refine and enhance your instructions for clarity. Here's a revised version:

---

# Solvation Meta Predictor

This repository contains code for predicting the aqueous solubility of organic molecules using machine learning models. The models and dataset are based on the research paper: [Predicting Aqueous Solubility of Organic Molecules Using Deep Learning Models with Varied Molecular Representations](https://pubs.acs.org/doi/full/10.1021/acsomega.2c00642).

## Usage

1. **Download Data**: Download the dataset from [this link](https://figshare.com/s/542fb80e65742746603c) and save it as `data.csv` in the `./data` folder.

2. **Generate Features**:
    - Generate Pybel coordinates and Molecular Dynamics (MDM) features by running `create_data.py` in the `./data` folder:
      ```sh
      cd ./data
      python create_data.py
      ```

3. **Train Models**:
    - To train the MDM model, run `train.py` in the `./mdm` folder:
      ```sh
      cd ../mdm
      python train.py
      ```
    - To train the GNN model, run `train.py` in the `./gnn` folder:
      ```sh
      cd ../gnn
      python train.py
      ```
    - To train the SMI model, run `train.py` in the `./smi` folder:
      ```sh
      cd ../smi
      python train.py
      ```

4. **Make Predictions**:
    - Use the `predict.ipynb` files in each model's folder to make predictions:
      ```sh
      cd ../mdm
      jupyter notebook predict.ipynb
      ```
      Repeat the above steps for the `gnn` and `smi` folders.

5. **Ensemble Models**:
    - To ensemble the models, run the following scripts:
      ```sh
      cd ../ensemble
      python CV.py
      python Optuna.py
      python KNN.py
      ```

6. **Compare Predictions**:
    - To compare predictions from individual models with ensemble methods, use the `ensemble_prediction.ipynb` notebook:
      ```sh
      jupyter notebook ensemble_prediction.ipynb
      ```

## Additional Information
For detailed instructions on how to run the models, featurize the data, and other specifics, please refer to the original research paper linked above. The methods and techniques described in the paper are critical for understanding and effectively using this repository.
