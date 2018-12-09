# -*- coding: utf-8 -*-
"""
Kaggle script for the competition Human protein Atlas
Python 3 virtual environment venv_pytorch/

@author: Yoann Pradat
"""

import os
import sys
import numpy as np 
import pandas as pd
import scipy
from scipy.stats import mode
from imageio import imread

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import PIL.Image as Image
from cnn_finetune import make_model
import xgboost as xgb

import gc
gc.enable()

from sklearn import metrics
import xgboost as xgb

print(sys.version)
print(os.getcwd())


"""
Functions used to train and validate models and to make predictions on the test set
"""

# =============================================================================
# Preprocessing function
# =============================================================================

def sigmoid(z):
    return 1/(1+np.exp(-z))

def default_loader(path):
    red    = Image.open(path + '_red.png')
    green  = Image.open(path + '_green.png')
    blue   = Image.open(path + '_blue.png')
    return Image.merge("RGB",(red,green,blue))

# torch Dataset object to load images and labels in separate files
class ImageFileList(torch.utils.data.Dataset):
    def __init__(self, root, list_imnames, list_targets, transform=None, loader=default_loader):
        self.root = root
        self.list_imnames = list_imnames
        self.list_targets = list_targets
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        imname, target = self.list_imnames[index], self.list_targets[index]
        img = self.loader(os.path.join(self.root,imname))
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.list_imnames)

# Normalize images 
# Mean and std vectors are standard for pretrained models

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = label_names[int(num)]
        row.loc[name] = 1
    return row

# =============================================================================
# Train and validation functions for CNN trained with SGD
# =============================================================================

def threshold_selection(probas, target, resolution=0.01):
    # In case output of CNN is not a proba
    if not probas.min()>=0 and probas.max()<=1:
        probas = sigmoid(probas)

    optimal_th = 0
    optimal_f1_score = 0
    for th in np.linspace(0, 1, 1/resolution+1):
        pred = np.where(probas > th, 1, 0)
        f1_score = metrics.f1_score(y_pred = pred, y_true=target)
        if f1_score > optimal_f1_score:
            optimal_th = th
            optimal_f1_score = f1_score
    return optimal_th, optimal_f1_score

def train(model, train_loader, batch_size, epoch, row_results, optimizer, criterion):
    model.train()

    loss_train = 0
    batches_output = []
    batches_target = []
    n_fit = len(train_loader.dataset)

    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.data.item()
                
        output = output.cpu().data.numpy()
        target = target.cpu().data.numpy()

        batches_output.append(output)
        batches_target.append(target)
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(data), n_fit, 100.* batch_idx / np.int(n_fit / batch_size), loss.data.item()))  

    
    output = np.concatenate(batches_output, axis=0)
    target = np.concatenate(batches_target, axis=0)
    probas = sigmoid(output.argmax(axis=1))
    print("Selecting optimal threshold for f1_score on train data...")
    threshold, _ = threshold_selection(probas, target, 0.01)
    print("done")
    pred = np.where(probas > threshold, 1, 0)
    precision, recall, f1_score, support = metrics.precision_recall_fscore_support(y_true=target, y_pred=pred)

    print('\nTrain Epoch: {} \tLoss: {:.3f}, precision: {:.3f}%, recall: {:.3f}%, f1_score: {:.3f}, support: {}/{} \
          support_pred {}/{}\n'.format(loss_train, precision[1], recall[1], f1_score[1], support[1], len(target),
                                       sum(pred), len(target)))
        
    row_results["threshold"] = threshold
    row_results["loss_train"] = loss_train
    row_results["accuracy_train"] = precision[1]
    row_results["recall_train"] = recall[1]
    row_results["f1_score_train"] = f1_score[1]
    row_results["support_train"] = support[1]

def validation(model, val_loader, row_results, criterion):
    model.eval()
    
    loss_val = 0
    threshold = row_results["threshold"]
    batches_pred = []
    batches_target = []

    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss_val += criterion(output, target).data.item()

        # get the probabilites above the threshold
        output = output.cpu().data.numpy().max(axis=1)
        probas = sigmoid(output)
        pred = np.where(probas > threshold, 1, 0)
        target = target.cpu().data.numpy()

        batches_pred.append(pred)
        batches_target.append(target)
        
    pred = np.concatenate(batches_pred, axis=0)
    target = np.concatenate(batches_target, axis=0)
    precision, recall, f1_score, support = metrics.precision_recall_fscore_support(y_true=target, y_pred=pred)

    print('Validation set: loss: {:.3f}, precision: {:.3f}%, recall: {:.3f}%, f1_score: {:.3f}, support: {}/{}, \
          support_pred {}/{}\n'.format(loss_val, precision, recall, f1_score, support, len(target), sum(pred),
                                       len(target)))

    row_results["loss_val"] = loss_val
    row_results["accuracy_val"] = precision[1]
    row_results["recall_val"] = recall[1]
    row_results["f1_score_val"] = f1_score[1]
    row_results["support_val"] = support[1]

# =============================================================================
# Train and validation functions for classifiers other than nn as last layer
# We load images by mini-batch, get a feature vector from the underlying CNN
# and store the new features in an array
# Do the same on validation set and then run a "classical" classifier
# =============================================================================

def train_batch(model, train_loader, batch_size, val_loader, row_results, name_clf, clf):
    model.train()
    
    batches_data_train, batches_data_val = [], []
    batches_target_train, batches_target_val = [], []
    
    print("Loading batches for training...")
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        
        # Compute output of pretrained model and store it in numpy format
        output = model(data)
        output = output.view(output.size()[0], -1).cpu().data.numpy()
        batches_data_train.append(output)
        
        # Store target in numpy format
        target = target.cpu().data.numpy()
        batches_target_train.append(target)
    print("done")
    
    print("Loading batches for validation...")
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        # Compute output of pretrained model and store it in numpy format
        output = model(data)
        output = output.view(output.size()[0], -1).cpu().data.numpy()
        batches_data_val.append(output)
        
        # Store target in numpy format
        target = target.cpu().data.numpy()
        batches_target_val.append(target)
    print("done")
        
    data_train = np.concatenate(batches_data_train, axis=0)
    target_train = np.concatenate(batches_target_train, axis=0)
    data_val = np.concatenate(batches_data_val, axis=0)
    target_val = np.concatenate(batches_target_val, axis=0)   

    print("shape data train %d, %d" % data_train.shape)
    print("shape data val %d, %d" % data_val.shape)

    if name_clf == "xgboost":
        clf.fit(data_train, target_train, eval_set=[(data_val, target_val)])
    else:
        clf.fit(data_train, target_train)

    # Optimal threshold and metric on training set
    probas = clf.predict_proba(data_train)[:,1]
    print("Selecting optimal threshold for f1_score on train data...")
    threshold, _ = threshold_selection(probas, target_train, 0.01)
    print("done")
    pred = np.where(probas > threshold, 1, 0)
    precision, recall, f1_score, support = metrics.precision_recall_fscore_support(y_true=target_train, y_pred=pred)

    print('Train: precision: {:.3f}%, recall: {:.3f}%, f1_score: {:.3f}, support: {}/{}, support_pred {}/{}\n'.format(
        precision[1], recall[1], f1_score[1], support[1], len(target_train), sum(pred), len(target_train)))
        
    row_results["threshold"] = threshold
    row_results["loss_train"] = 0
    row_results["accuracy_train"] = precision[1]
    row_results["recall_train"] = recall[1]
    row_results["f1_score_train"] = f1_score[1]
    row_results["support_train"] = support[1]

    # Metrics on validation set
    probas = clf.predict_proba(data_val)[:,1]
    pred = np.where(probas > threshold, 1, 0)
    precision, recall, f1_score, support = metrics.precision_recall_fscore_support(y_true=target_val, y_pred=pred)

    print('Val: precision: {:.3f}%, recall: {:.3f}%, f1_score: {:.3f}, support: {}/{}, support_pred {}/{}\n'.format(
        precision[1], recall[1], f1_score[1], support[1], len(target_val), sum(pred), len(target_val)))
        
    row_results["loss_val"] = 0
    row_results["accuracy_val"] = precision[1]
    row_results["recall_val"] = recall[1]
    row_results["f1_score_val"] = f1_score[1]
    row_results["support_val"] = support[1]


# =============================================================================
# Functions for loading test images and making predictions
# =============================================================================

def make_predictions(model, test_path, outfile, df_test_lab, threshold, clf=None):
    output_file = open(outfile, "w")
    output_file.write("Id,Predicted\n")
    print("Making predictions on the test set...")
    for Id in tqdm(df_test_lab.Id):
        data = data_transforms(default_loader(os.path.join(test_path, Id)))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        
        if use_cuda:
            data = data.cuda()

        output = model(data)
        
        if clf == None:
            output = output.cpu().data.numpy().max(axis=1)
            proba = sigmoid(output)
            pred = np.where(proba > threshold, 1, 0)
        else:
            output = output.view(output.size()[0], -1).cpu().data.numpy()
            proba = clf.predict_proba(output).flatten()
            pred = np.where(proba > threshold, 1, 0)[1]

        output_file.write("%s,%d\n" % (Id, pred))
    output_file.close()
    print("Succesfully wrote " + outfile + ', you can upload this file to the kaggle competition website\n')

"""
Load target for train
"""

df_train_lab = pd.read_csv("data/train.csv")
df_test_lab = pd.read_csv("data/sample_submission.csv")
n_train = df_train_lab.shape[0]
n_test = df_test_lab.shape[0]
print("Number of train and test samples: %d and %d" % (n_train, n_test))

label_names = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

reverse_label_names = {v:k for k,v in label_names.items()}


# Create dummy columns for each label in df_train_lab
for v in label_names.values():
    df_train_lab[v] = 0 
df_train_lab = df_train_lab.apply(fill_targets, axis=1)
df_dummy_lab = df_train_lab.iloc[:, 2:]


"""
Toy models

In a first approach, solve 28 binary classification problems. For each binary classification we want
to get the highest possible F1-Score
"""

label = "Peroxisomes"

# Training settings
train_path = "data/train/"
test_path = "data/test/"
experiment = "experiment/"
if not os.path.isdir(experiment):
    os.makedirs(experiment)

n_classes = 2
batch_size = 64
nepoch = 5
momentum = 0.9
lr = 0.01
log_interval = 1
seed = 95
use_cuda = torch.cuda.is_available()
torch.manual_seed(seed)

# Separate data into training and validation
n_train = 100
n_fit = np.int(0.8*n_train)
n_val = n_train - n_fit

# Create training and validation loaders that can load images in mini-batches
np.random.seed(seed)
rand_rows = np.random.permutation(n_train)
fit_data = ImageFileList(root=train_path, list_imnames=df_train_lab.Id[rand_rows[:n_fit]].values, 
                         list_targets=df_dummy_lab.loc[rand_rows[:n_fit], label].values, transform=data_transforms)
val_data = ImageFileList(root=train_path, list_imnames=df_train_lab.Id[rand_rows[n_fit:]].values, 
                         list_targets=df_dummy_lab.loc[rand_rows[n_fit:], label].values, transform=data_transforms)

train_loader = torch.utils.data.DataLoader(fit_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

# =============================================================================
# Classifiers to use on feature vectors extracted from CNN
# =============================================================================
classifiers = []

# Classifier xgboost
params_xgb = {'booster': 'gbtree',
              'n_estimators': 50,
              'max_depth':15,
              'min_child_weight': 1,
              'learning_rate':.1,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'seed': 1,
              'num_boost_round' : 35,
              'early_stopping_rounds' : 5,
              'objective': 'binary:logistic',
              'eval_metric': ['logloss'], 
              'silent': 1}

clf_xgb = xgb.XGBClassifier(**params_xgb)
classifiers.append(("xgboost", clf_xgb))

# Default classifier i.e fully connected neural network
# Number of layers, activation function, etc. depend on the pretrained model

classifiers.append(("default", None))

# =============================================================================
# Run pretrained models and classifiers defined above
# =============================================================================

# Because CUDA memory is limited (12GB), we can't update all parameters of pretrained models
def choose_fixed_layers(pretrained, model):
    if pretrained == "densenet121":
        # Let weights of first 2 layers unchanged
        for param in list(model.children())[0][0].parameters():
            param.requires_grad = False
        for param in list(model.children())[0][1].parameters():
            param.requires_grad = False

        # _Dense Block is composed of 6 dense layers
        # Let weights of first 3 layers unchanged
        for i in range(3):
            for param in list(model.children())[0][4][i].parameters():
                param.requires_grad = False

    elif pretrained == "inception_v3":
        # First 7 layers of inception_v3 are BasicConv2d and MaxPool2D
        # Next 4 are InceptionA layers
        for i in range(11):
            for param in list(model.children())[0][i].parameters():
                param.requires_grad = False

    elif pretrained == "se_resnet50":
        # CUDA memory is enough to update all parameters
        pass
    
    elif pretrained == "inceptionresnetv2":
        # First 15 layers, only update params of last one
        for i in range(14):
            for param in list(model.children())[0][i].parameters():
                param.requires_grad = False


pretrained_models = ["se_resnet50", "densenet121"]
validation_results = pd.DataFrame(columns=["pretrained", "classifier", "epoch", "threshold", "loss_train", 
                                           "accuracy_train", "recall_train", "support_train", "f1_score_train", 
                                           "loss_val", "accuracy_val", "recall_val", "support_val", "f1_score_val"])

for pretrained in pretrained_models:
    if not os.path.isdir("experiment/%s" % pretrained):
        os.mkdir("experiment/%s" % pretrained)

    for name_clf, clf in classifiers:
        model = make_model(pretrained, num_classes=n_classes, pretrained=True, input_size=(224, 224))
        if name_clf == "default":
            choose_fixed_layers(pretrained, model)
        else:
            removed = list(model.children())[:-1]
            model= torch.nn.Sequential(*removed)

        # Use default CNN. Train several epochs
        if clf == None:
            print("\n")
            print("#"*40)
            print("Model specifications")
            print("Pretrained: %s" % pretrained)
            print("Classifier: %s" % name_clf)
            print("SGD learning rate: %s" % lr)
            if use_cuda:
                print('Using GPU')
                model.cuda()
            else:
                print('Using CPU')
            print("#"*40)
            print("\n")
    
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Train the model on the selected epoch on the train dataset and then compute validation accuracy on val dataset
            for epoch in range(1, nepoch + 1):
                row_results = {"pretrained": pretrained, "classifier": name_clf, "epoch": epoch}

                train(model, train_loader, batch_size, epoch, row_results, optimizer, criterion)
                validation(model, val_loader, row_results, criterion)

                validation_results = validation_results.append(row_results, ignore_index=True)
                validation_results.to_csv("../working/experiment/validation_results.csv")

                # Make predictions on test images
                model.eval()
                if use_cuda:
                    model.cuda()

                outfile = os.path.join(experiment, "s/kaggle_%s_%s_epoch%d.csv" % (pretrained, pretrained, name_clf,
                                                                                   epoch))
                make_predictions(model, test_path, outfile, df_test_lab, row_results["threshold"])
         
        # Use pretrained model to extract feature vector then use classifier
        else:
            print("\n")
            print("#"*40)
            print("Model specifications")
            print("Pretrained: %s" % pretrained)
            print("Classifier: %s" % name_clf)
            if use_cuda:
                print('Using GPU')
                model.cuda()
            else:
                print('Using CPU')
            print("#"*40)
            print("\n")

            row_results = {"pretrained": pretrained, "classifier": name_clf, "epoch": 1}
            train_batch(model, train_loader, batch_size, val_loader, row_results, name_clf, clf)

            # Make predictions on test images
            model.eval()
            if use_cuda:
                model.cuda()
            
            outfile = os.path.join(experiment, "%s/kaggle_%s_%s_epoch%d.csv" % (pretrained, pretrained, name_clf, 1))
            make_predictions(model, test_path, outfile, df_test_lab, row_results["threshold"], clf)

        # Free cuda memory
        del model ; gc.collect()
        torch.cuda.empty_cache()


######################################################
# Code that mixes predictions of the differents models
# One technique is used majority vote to classify
######################################################

path = experiment

# Read output files
vals = validation_results
output = pd.read_csv(path + "se_resnet50/kaggle_se_resnet50_default_epoch1.csv", usecols=["Id"])
        
# Select predictions of models having validation performance above threshold
outputs_model = pd.DataFrame.copy(output)
del outputs_model["Id"]
for pretrained in vals.pretrained.unique():
    for classifier in vals.classifier.unique():
        for epoch in vals.epoch.unique():
            mask = (vals.pretrained == pretrained) & (vals.classifier == classifier) & (vals.epoch == epoch)
            name = "%s_%s_%s_%s" % (label, pretrained, classifier, epoch)
            outfile = "%s/kaggle_%s_%s_epoch%d.csv" % (pretrained, pretrained, classifier, epoch)
            out = pd.read_csv(path  + outfile)
            outputs_model.loc[:, name] = out["Predicted"]
                
output.loc[:, label] = outputs_model.apply(lambda x: mode(x)[0][0], axis=1)
output.to_csv("pred_%s.csv" % label, index=False)
