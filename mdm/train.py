#import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from sklearn.model_selection import KFold
from rdkit import Chem
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import json
import mdm_model
import config
import mdm_utils
import datetime

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import RandomNormal, RandomUniform


def run():
    
    to_remove = [ 'cas', 'ref', 'temp','inchi']
 
    #loading training, test and validation data

    train = pd.read_csv(config.data_dir+"train.csv")
    val = pd.read_csv(config.data_dir+"val.csv")
    test = pd.read_csv(config.data_dir+"test.csv")

    #dropping unnecessary columns

    train = train.drop(to_remove, axis=1)
    val = val.drop(to_remove, axis=1)
    test = test.drop(to_remove, axis=1)
    
    #checking for duplicates in Datasets

    mdm_utils.check_duplicates(train,val,test)

    trainx = train
    valx = val
    testx = test

    to_drop = ['log_sol', 'smiles']

    #transform data
    x_train,y_train, x_test, y_test, x_val, y_val, sc = mdm_utils.get_transformed_data(train   = trainx, 
                                                                             val     = valx, 
                                                                             test    = testx, 
                                                                             to_drop = to_drop, 
                                                                             y       = "log_sol")

    #create model
    model = mdm_model.create_model(x_train.shape[1])
    
    #define callbacks for early stopping and saving the best model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=config.patience)
    if os.path.exists(config.best_model):
        os.remove(config.best_model)
    mc = ModelCheckpoint(f'{config.best_model}', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    #train model
    result = model.fit(x_train, y_train, batch_size = config.batch_size, epochs = config.max_epochs,
              verbose = 2, validation_data = (x_val,y_val), callbacks = [es,mc])


    print(f"training completed at {datetime.datetime.now()}")

    
if __name__ == "__main__":
    run()
