import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

class ModelUtils:
    
    @staticmethod
    def plot_cm(y_true, y_pred, class_names,save_path, figsize=(8,8)):
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        # cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
        cm = pd.DataFrame(cm, index=[i for i in class_names],
                    columns = [i for i in class_names])
        
        cm.index.name = 'True label'
        cm.columns.name = 'Predicted label'
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=annot, fmt='', ax=ax)
        plt.savefig(save_path)

    @staticmethod
    def plot_loss_accuracy(save_path,history, figsize=(8, 8)):
        fig, ax = plt.subplots(1, 2, figsize= figsize)
        ax[0].plot(history.history['loss'], label='train')
        ax[0].plot(history.history['val_loss'], label='validation')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Loss vs Epoch')
        ax[0].legend()
        ax[1].plot(history.history['accuracy'], label='train')
        ax[1].plot(history.history['val_accuracy'], label='validation')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_title('Accuracy vs Epoch')
        ax[1].legend()
        plt.savefig(save_path)
        plt.show()

    # function for scoring roc auc score for multi-class
    @staticmethod
    def multiclass_roc_auc_score(y_true, model_predicted_label,class_names,save_path, average="macro", figsize = (8, 8)):
        fig, c_ax = plt.subplots(1,1, figsize =figsize)
        lb = LabelBinarizer()
        lb.fit(y_true)
        y_test = lb.transform(y_true)
        y_pred = lb.transform(model_predicted_label)
        
        for (idx, c_label) in enumerate(class_names):
            fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
            c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
        c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
        
        c_ax.legend()
        c_ax.set_xlabel('False Positive Rate')
        c_ax.set_ylabel('True Positive Rate')
        plt.savefig(save_path)
        plt.show()
        return roc_auc_score(y_test, y_pred, average=average)