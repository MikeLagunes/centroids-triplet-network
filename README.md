# IROS 2020 - Centroids  Triplet  Network  and Temporally-Consistent  Embeddings  for In-Situ  Object  Recognition.

This website contains the PyTorch implemetation for the Centrois Triplet Network (CTN), as well as the links for downloaded the in-situ househol and CORe50 datasets used on the paper. 


# Training the model

To train the model use:

```
python 3 train/train_ctn.py
```

The training script finetunes a ResNet-50 backbone network, pre-trained with Imagenet, and logs the training into a txt file. Datasets can be downloaded from:

[In-Situ Household](https://drive.google.com/file/d/17qKY2QTtrA17jF3jhaL2SKS1ZHuzJcbY/view?usp=sharing)

[Core50](https://drive.google.com/file/d/1Hr9wnV9tYZb6KTfoHWBOasid7fGJB7xw/view?usp=sharing)

If you find this work useful, please cite it as follows:

```
bibtex
```