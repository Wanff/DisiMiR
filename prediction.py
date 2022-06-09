from re import M
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

from matplotlib import pyplot as plt
from scipy.stats import hypergeom

def load_train_test_split(miRNA_data = None, random_state = None, miRNA_data_path = None):
    """
    Creates a training/testing split from a pandas Dataframe or csv file

    Input:
        miRNA_data: pd.Dataframe
        random_state: int
        miRNA_data_path: string
    Output:
        train/test split
    """

    if miRNA_data_path is not None:
        miRNA_data = pd.read_csv(miRNA_data_path)

    X = miRNA_data[['Disease_Influence', 'Network_Influence', 'Conservation', 'Num_Targets']]
    y = miRNA_data[['Causality']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test

def stratified_random_split(miRNA_data = None, random_state = None, miRNA_data_path = None, num_splits = 3):
    if miRNA_data_path is not None:
        miRNA_data = pd.read_csv(miRNA_data_path)

    X = miRNA_data[['Disease_Influence', 'Network_Influence', 'Conservation', 'Num_Targets']]
    y = miRNA_data[['Causality']]

    skf = StratifiedKFold(n_splits=num_splits)

    train_indices = [split[0] for split in list(skf.split(X, y))]
    test_indices = [split[1] for split in list(skf.split(X, y))]
    
    return train_indices, test_indices, X, y

class AdaBoostEnsembleModel:
    def __init__(self, members):
        self.members = members
    
    def predict(self, X_test):
        # make predictions
        yhats = [model.predict(X_test) for model in self.members]
        yhats = np.array(yhats)

        # sum across ensemble members
        summed = np.sum(yhats, axis=0)

        # argmax across classes
        result = [True if sum >= 3 else False for sum in summed ]
        return result
    
    def predict_proba(self, X_test):
        #ada.classes_ = [False, True]

        # make predictions
        yhats = [model.predict_proba(X_test) for model in self.members]
        yhats = np.array(yhats)

        # sum across ensemble members
        summed = np.sum(yhats, axis=0) / len(self.members)

        #y_pred[:,1] just gets the positive class according to the ada.classes_
        return summed
    
    def feature_importances(self):
        feature_importances = []
        for model in self.members:
            feature_importances.append(model.feature_importances_)

        return np.mean(feature_importances, axis = 0)


def make_ensemble(X_train, y_train):
    kf = KFold(n_splits=5)

    members = []
    npX = np.asarray(X_train)
    npY = np.asarray(y_train)
    for train_index, test_index in kf.split(npX):
        cv_x_train = npX[train_index]
        cv_y_train = npY[train_index]
        
        ada = AdaBoostClassifier(random_state=None, n_estimators=1500)

        ada.fit(cv_x_train,cv_y_train.ravel())
        members.append(ada)
    
    ensemble = AdaBoostEnsembleModel(members)
    return ensemble

def plot_auc(predictions, auc_label = "Disease", filename = 'auc', save_path = "disease_data/"):
    """
    predictions should be a Pandas dataframe wtih the predictions for a disease. 

    Plots AUC curve for one disease 

    If you don't want to save the AUC image, set save_path to None

    Set auc_label to the name of the disease that you're plotting 
    """
    y_test = predictions["HMDD_Class"]
    y_pred = predictions["Average_Prob"]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    rocauc = roc_auc_score(y_test, y_pred)

    rgb = np.random.rand(3,)

    plt.plot(fpr, tpr, c = rgb, label = auc_label + ' AUC = %0.2f' % rocauc)

    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    if save_path is not None:
        plt.savefig(save_path+filename+'.png', bbox_inches='tight')

    plt.show()

def predict_disease_causality(X_train, X_test, y_train, y_test, use_pretrained_model = False, miRNA_data = None, return_value = None):
    if use_pretrained_model is False:
        # model = make_ensemble(X_train, y_train)
        model = AdaBoostClassifier(random_state=None, n_estimators=1500)
        model.fit(np.asarray(X_train), np.asarray(y_train).ravel())

        y_pred = model.predict_proba(X_test)[:,1]
    else:
        model = pd.read_pickle('cancer_aggregate_model.pickle')

        y_pred = model.predict_proba(X_test)[:,1]

    if return_value == "auc":
        rocauc = roc_auc_score(y_test, y_pred)
        return rocauc
    
    elif return_value == "false_positives":
        #false positives in a single run
        optimal_cutoff = find_threshold_least_errors(y_test, y_pred, false_pos_weight= 1, false_neg_weight=1)

        false_positives = []
        for pred, actual in zip(y_pred, y_test.reset_index().values.tolist()): #https://note.nkmk.me/en/python-pandas-list/
            if pred > optimal_cutoff and actual[1] == False:
                false_positives.append(actual[0])
    
        return false_positives
    
    elif return_value == "metrics_and_predictions":
        rocauc = roc_auc_score(y_test, y_pred)

        causal_count = miRNA_data['Causality'].sum()
        population_size = len(miRNA_data.index)

        optimal_cutoff = find_threshold_least_errors(y_test, y_pred, false_pos_weight= 1, false_neg_weight=1)

        y_pred_class = []
        for pred in y_pred:
            if pred > optimal_cutoff:
                y_pred_class.append(1)
            else:
                y_pred_class.append(0)
        
        CM = confusion_matrix(y_test, y_pred_class)

        p_value = hypergeom.sf(CM[1][1]-1, population_size, causal_count, CM[1][1]+CM[0][1])

        # feature_importances = model.feature_importances()
        feature_importances = model.feature_importances_


        false_positives = []
        true_positives = []

        predictions = [] #(prob, mir, class)
        for pred, actual in zip(y_pred, y_test.reset_index().values.tolist()): #https://note.nkmk.me/en/python-pandas-list/
            if pred > optimal_cutoff and actual[1] == False:
                false_positives.append(actual[0])
            if pred > optimal_cutoff and actual[1] == True:
                true_positives.append(actual[0])
            
            predictions.append((pred, actual[0], actual[1])) #prediction, mir name, label

        # positives_in_test = true_positives + false_positives

        return rocauc, CM, p_value, feature_importances, false_positives, predictions, optimal_cutoff

def find_threshold_least_errors(y_test, y_pred, false_pos_weight = 1, false_neg_weight = 1):
    __, __, thresholds = roc_curve(y_test, y_pred)
    # print(len(thresholds))

    threshold_to_errors_dict = {}
    for threshold in thresholds:
        y_pred_class = []
        for pred in y_pred:
            if pred > threshold:
                y_pred_class.append(1)
            else:
                y_pred_class.append(0)
        
        CM = confusion_matrix(y_test, y_pred_class)

        threshold_to_errors_dict[threshold] = CM[0][1] * false_pos_weight + CM[1][0] * false_neg_weight if CM[0][1] > 10 else np.inf

    min_error = min(threshold_to_errors_dict.values())

    for threshold, error in threshold_to_errors_dict.items():
        if error == min_error:
            return threshold