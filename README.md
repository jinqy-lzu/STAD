
## Identification and Validation of the Molecular Subgroups of Ovarian Cancer and Its Related Cancers using a Self-supervised Machine Learning Method Based on Disulfidptosis-Related Genes

## Requirements
- Python3
- PyTorch (2.4.1)
- torchvision (0.19.1)
- numpy (1.24.4)

## Cluster
Run ```./CULSTER/clustering-model.py```

## Training classification model
Run ```./CLASSIFICATION/CNN+BiLSTM.py```, training classification models.

```
python ./CLASSIFICATION/CNN+BiLSTM.py  --train_file_path ./STAD_train_exp_combat1.csv --test_file_path ./STAD_test_exp_combat1.csv  --best_model_path ./result/CNN+BiLSTM --batch_size 64 --lr 0.05 --trail 1
```

