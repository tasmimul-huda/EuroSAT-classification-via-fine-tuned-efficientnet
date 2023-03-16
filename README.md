# EuroSAT-classification-via-fine-tuned-efficientnet

## How to run
- create a virtualenv: `virtualenv env`
- activate env: `env\Scripts\activate`
- install  dependencies: `pip install requirements.txt`
- configure the config.py. specify necessary datapath and model_type and all figures path
- run main.py: `python main.py`

**There are two baseline models. I played a bitwith the architecture and the hyperparameters for optimization**

## Output--> BaselineCNN ---without augmentation
**Loss Accuracy Curve For Baseline CNN**
![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/loss_accuracy_curve.png?raw=true)

<!--**Confusion Matrix For Baseline CNN**
![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/confusion_matrix.png?raw=true)

**Roc Auc For Baseline CNN**
![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/roc_auc_curve.png?raw=true)-->

** In case of efficientnet without augmentation I freezes some layers of the efficient_model according to the following**
`for layer in base_model.layers[:30]:
            layer.trainable = False`
## Output--> EfficientNet B1--without augmentation
### Efficientnet B1

> train_loss: 0.03542402386665344 || train_accuracy:0.9920105934143066
> val_loss: 0.3068503737449646: || val_acc: 0.903333306312561
> test_loss:0.26120659708976746 || test_acc: 0.9103703498840332

**Loss Accuracy Curve For Efficientnet B1**

![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/efficientnetB1_loss_accuracy_curve.png?raw=true)

[Confusion Matrix](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/efficientnetB1_confusion_matrix.png) || [ROC AUC](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/efficientnetB1_roc_auc_curve.png)

<!--**Confusion Matrix For Efficientnet B1**
![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/efficientnetB1_confusion_matrix.png?raw=true)

**Roc Auc For Efficientnet B1**
![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/efficientnetB1_roc_auc_curve.png?raw=true)-->

## Output--> EfficientNet B2--without augmentation
### Efficientnet B2

> train_loss:0.010462253354489803:: train_accuracy: 0.9976719617843628
> val_loss:0.23319227993488312:: val_acc: 0.9301851987838745
> test_loss:0.20635396242141724:: test_acc: 0.9351851940155029

**Loss Accuracy Curve For Efficientnet B2**

![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/efficientnetB1_loss_accuracy_curve.png?raw=true)

<!--**Confusion Matrix For Efficientnet B2**
![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/efficientnetB1_confusion_matrix.png?raw=true)

**Roc Auc For Efficientnet B2**
![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/efficientnetB1_roc_auc_curve.png?raw=true)-->


** In case of efficientnet using augmentation I freezes some layers of the efficient_model according to the following**
`for layer in self.base_model.layers[:40]:  #-20
            layer.trainable = False`

## Output--> EfficientNet B2--using augmentation
### Efficientnet B2

> train_loss:0.19670043885707855:: train_accuracy: 0.931587278842926
> val_loss:0.2384759485721588:: val_acc: 0.9235185384750366
> test_loss:0.22396968305110931:: test_acc: 0.9233333468437195

**Loss Accuracy Curve For Efficientnet B2**

![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/efficientnetB2_&_augmentation_loss_acc_curve.png?raw=true)



**The confusion matrix and Roc curve for all models are in figures folder.**


