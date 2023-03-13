import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

class ModelUtils:
    
    @staticmethod
    def plot_cm(y_true, y_pred, class_names,save_path, figsize=(10,10)):
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
    def plot_confusion_matrix(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='g')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        plt.show()
        
    

    @staticmethod
    def plot_loss_accuracy(save_path,history):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
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
    def multiclass_roc_auc_score(y_true, model_predicted_label,class_names,save_path, average="macro"):
        fig, c_ax = plt.subplots(1,1, figsize = (16, 16))
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






'''
## confusion matrix
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# create a dataset of tuples (image, label) where label is an integer
test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ],
        [1, 2, 0]
    )
)

# define a dictionary that maps integer labels to string labels
label_map = {0: "cat", 1: "dog", 2: "bird"}

# obtain the true labels and predicted labels from your test dataset
true_labels = []
predicted_labels = []
for image, label in test_dataset.as_numpy_iterator():
    true_labels.append(label_map[label])
    # here you would have code to predict the label from the image
    predicted_labels.append(label_map[np.argmax(image)])

# compute the confusion matrix using scikit-learn's confusion_matrix() function
confusion_mtx = confusion_matrix(true_labels, predicted_labels)

# plot the confusion matrix using matplotlib's imshow() function
fig, ax = plt.subplots()
im = ax.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_xticks(np.arange(len(label_map)))
ax.set_yticks(np.arange(len(label_map)))
ax.set_xticklabels(list(label_map.values()))
ax.set_yticklabels(list(label_map.values()))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(label_map)):
    for j in range(len(label_map)):
        text = ax.text(j, i, confusion_mtx[i, j],
                       ha="center", va="center", color="w")
plt.show()


'''


'''

    report = classification_report(true_labels, predicted_label, target_names=class_names)
    report_dict = classification_report(true_labels, predicted_label, target_names=class_names, output_dict=True, zero_division=0)
    
    # Convert the report to a dictionary

    # Create a bar plot of the report
    print(report)
    print(report_dict.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(report_dict['macro avg'].keys(), report_dict['macro avg'].values())
    ax.set_ylim(0, 1)
    ax.set_title('Classification Report')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')

    # Save the plot as an image
    plt.savefig('classification_report.png')
    plt.show()
    
    
    # print(report_dict.keys())
    # for key in report_dict.keys():
    #     if isinstance(report_dict[key], dict):
    #         print(f'{key}: {report_dict[key]["precision"]}')
                    
    # plt.bar(report_dict.keys(), [report_dict[key]['precision'] for key in report_dict.keys()])
    # plt.title('Classification Report')
    # plt.xlabel('Class')
    # plt.ylabel('Precision')
    # plt.show()
    
    # fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

    # metrics = ['precision', 'recall', 'f1-score']
    # for i, metric in enumerate(metrics):
    #     # print(metric)
    #     axs[i].bar(report_dict.keys(), [report_dict[key][metric] for key in list(report_dict.keys())])
    #     axs[i].set_title(metric.title())
    #     axs[i].set_xlabel('Class')
        
    # axs[0].set_ylabel('Score')
    # plt.show()


'''