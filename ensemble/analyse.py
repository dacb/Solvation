import shap
import numpy as np
import os
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
import mdm

def load_data():
    print("Loading data...")
    mdm_x_test = np.loadtxt('/home/fostooq/solubility-prediction-paper1/input/x_test.txt')
    mdm_y_test = np.loadtxt('/home/fostooq/solubility-prediction-paper1/input/y_test.txt')
    return mdm_x_test, mdm_y_test

def load_keras_model():
    print("Loading model...")
    return load_model('./mdm/mdm_best.h5')  # Update path as needed

def compute_shap_values(model, x_train):
    print("Creating SHAP DeepExplainer...")
    explainer = shap.DeepExplainer(model, x_train)
    print("Computing SHAP values...")
    return explainer.shap_values(x_train)

def save_shap_values(shap_values):
    print("Saving SHAP values...")
    with open('shap_values.pkl', 'wb') as f:
        pickle.dump(shap_values, f)

def generate_and_save_plot(shap_values, x_train):
    print("Generating plot...")
    shap.summary_plot(shap_values[0], x_train, plot_type="bar")
    plt.savefig('shap_summary_plot.png')
    print("Plot saved as 'shap_summary_plot.png'.")

def main():
    if not os.path.exists('output'):
        os.makedirs('output')
    os.chdir('output')
    
    x_test, y_test = load_data()
    model = load_keras_model()
    shap_values = compute_shap_values(model, x_test)
    
    save_shap_values(shap_values)
    generate_and_save_plot(shap_values, x_test)

if __name__ == '__main__':
    main()