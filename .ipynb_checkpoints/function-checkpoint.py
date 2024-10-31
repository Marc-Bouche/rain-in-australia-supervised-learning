import string
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import math

#---------------------------------------------------------------------------------------------------#

# Function 1
# This function calculate metrics from a confusion matrix 
# (Grid Search - Hyperparameters | Step 1: Gather Model Performances Results)

def evaluate_model(conf_matrix, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1