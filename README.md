# EuroSAT-classification-via-fine-tuned-efficientnet

## How to run
- create a virtualenv: `virtualenv env`
- activate env: `env\Scripts\activate`
- install  dependencies: `pip install requirements.txt`
- configure the config.py. specify necessary datapath
- run main.py: `python main.py`

## Output--> BaselineCNN ---without augmentation
**Loss Accuracy Curve For Baseline CNN**
![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/loss_accuracy_curve.png?raw=true)

**Confusion Matrix For Baseline CNN**
![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/confusion_matrix.png?raw=true)

**Roc Auc For Baseline CNN**
![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/roc_auc_curve.png?raw=true)

## Output--> EfficientNet B1--without augmentation
### Efficientnet B1

> Train Loss: 0.03542402386665344 || Train Accuracy:0.9920105934143066
> Val Loss: 0.3068503737449646: || Val Accuracy: 0.903333306312561
> Test Loss:0.26120659708976746 || Test Accuracy: 0.9103703498840332

**Loss Accuracy Curve For Efficientnet B1**

![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/efficientnetB1_loss_accuracy_curve.png?raw=true)

**Confusion Matrix For Efficientnet B1**
![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/efficientnetB1_confusion_matrix.png?raw=true)

**Roc Auc For Efficientnet B1**
![alt text](https://github.com/tasmimul-huda/EuroSAT-classification-via-fine-tuned-efficientnet/blob/main/Figures/efficientnetB1_roc_auc_curve.png?raw=true)

