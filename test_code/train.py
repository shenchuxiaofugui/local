
# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision.datasets import DatasetFolder
from torchvision.models import resnet50

# This is for the progress bar.
from tqdm.auto import tqdm
import torchmetrics
import pickle

import matplotlib.pyplot as plt
import time
import os
from dataload import CreateNiiDataset1

from model import Classifier1


# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
train_tfm1 = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((224, 224)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])


# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
# batch_size = 200

# Construct datasets.
custom_data = CreateNiiDataset1('test', ['T1'], use_roi=True)
custom_data.select_roi(3)

# with open('1.pkl', 'wb') as file:
#     pickle.dump(custom_data, file)
#     file.close()

# with open("1.pkl", 'rb') as file:
#     custom_data = pickle.loads(file.read())
#     file.close()

train_size = round(len(custom_data) * 0.8)
valid_size = round(len(custom_data) * 0.2)
print(train_size, valid_size, len(custom_data))
train_data, validate_dataset = torch.utils.data.random_split(custom_data, [train_size, valid_size])
print('successful again')

# for i in train_data.fin_data:
#     print(i.shape)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(train_data, batch_size=10, shuffle=True)

# "cuda" only when GPUs are available.
# "cuda" only when GPUs are available.
start = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"

# losses
train_losses = []
train_acces = []
valid_losses = []
valid_acces = []

# Initialize a model, and put it on the device specified.
model = Classifier1()
model = nn.DataParallel(model)
model = model.to(device)
model.device = device

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-3)

# The number of training epochs.
n_epochs = 20

# Whether to do semi-supervised learning.
do_semi = False
train_auc = torchmetrics.AUC()
val_auc = torchmetrics.AUC()

for epoch in range(n_epochs):
    # ---------- TODO ----------
    # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
    # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    # Iterate the training set by batches.
    for batch in tqdm(train_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        #acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        print(logits, labels.to(device))
        train_auc.update(logits, labels.to(device))

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        #train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    #train_acc = sum(train_accs) / len(train_accs)
    print(train_auc)
    train_acc = train_auc.compute()

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    train_losses.append(train_loss)
    train_acc = train_acc.cpu().item()
    train_acces.append(train_acc)
    train_auc.reset()

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        #acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        val_auc.update((logits.argmax(dim=-1), labels.to(device)))


        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        #valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    #valid_acc = sum(valid_accs) / len(valid_accs)
    valid_acc = val_auc.compute()

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    valid_losses.append(valid_loss)
    valid_acc = valid_acc.cpu().item()
    valid_acces.append(valid_acc)
    val_auc.reset()

torch.save(model,'net1.pkl')
end = time.time()
print(f"所用时间：{(end - start) / 60}分钟")
plt.title("loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(range(epoch+1), train_losses)
plt.plot(range(epoch+1), valid_losses)
plt.legend(["loss", "valid_loss"])
plt.savefig("out/T1_1_loss.jpg",dpi=300)
plt.show

plt.clf()
plt.title("auc")
plt.xlabel("epoch")
plt.ylabel("auc")
plt.plot(range(epoch+1), train_acces)
plt.plot(range(epoch+1), valid_acces)
plt.legend(["acc", "valid_auc"])
plt.savefig("out/T1_1_auc.jpg", dpi=300)
plt.show

#torch.save(model,'net1.pkl')
