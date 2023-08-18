#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
data = pd.read_csv('./dataset/stanced_data.csv')


# In[2]:


import ast

def parse_feature_string(feature_str):
    # Replace newline characters with spaces
    feature_str = feature_str.replace('\n', ' ')
    # print(feature_str)
    # Split the string by spaces, filter out empty strings, and then join with commas
    modified_str = ','.join(filter(None, feature_str.split(' ')))
    # print(modified_str)
    # Use ast.literal_eval to convert the modified string to a list
    float_list = ast.literal_eval('[' + modified_str + ']')
    
#    Convert scientific notation strings to float
    float_values = [val for val in float_list]
    # print(float_values)
    # Round to 8 decimal places
    # rounded_values = [round(val, 8) for val in float_values]
    
    return float_values


# In[8]:


# d = data[5:7]
parsed_features = data['style_feature'].apply(parse_feature_string)

# Convert the parsed features (list of lists) to a PyTorch tensor
tensor_style = torch.tensor(parsed_features.tolist(), dtype=torch.float64).squeeze(1)

# parsed_features,
parsed_features, tensor_style


# In[22]:


tensor_style[0][1].item()


# In[11]:


print(tensor_style.shape)
print(parsed_features.shape)


# In[23]:


one_hot_predictions = pd.get_dummies(data['predictions'])

# Convert the one-hot encoded DataFrame to a tensor
tensor_shance = torch.tensor(one_hot_predictions.values, dtype=torch.float64)
tensor_shance.shape


# In[13]:


#concat the style and stance feature tensor
feature_tensor = torch.cat((tensor_style, tensor_shance), dim=1)
print(feature_tensor.shape)
print(feature_tensor.dtype)


# In[24]:


# Convert feature_tensor to float32 dtype
feature_tensor = feature_tensor.to(dtype=torch.float64)

feature_tensor.dtype


# In[17]:


feature_tensor[0][1].item()


# In[25]:



# Convert to PyTorch tensors
X = feature_tensor
y = torch.tensor(data['label'].values, dtype=torch.float64)

# Split into training and validation sets
train_size = int(0.8 * len(X))
val_size = len(X) - train_size

X_train, X_val = torch.split(X, [train_size, val_size])
y_train, y_val = torch.split(y, [train_size, val_size])

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=128)

X.shape, y.shape
X.dtype, y.dtype


# In[26]:


import torch.nn as nn
import torch.optim as optim

# Hyperparameters
N_EMB = 64
HIDDEN_DIM = 64
OUTPUT_DIM = 1
DROPOUT = 0.2

# Define the modified model
class ModifiedModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(ModifiedModel, self).__init__()
        
        self.fc_input = nn.Linear(input_dim, N_EMB)
        self.lstm = nn.LSTM(N_EMB, hidden_dim, bidirectional=True, batch_first=True)
        self.mha = nn.MultiheadAttention(2*hidden_dim, num_heads=8)
        self.fc1 = nn.Linear(2*hidden_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.fc_input(text)
        # print(embedded.shape)
        lstm_out, _ = self.lstm(embedded.unsqueeze(1))
        # print(lstm_out.shape)
        attn_output, _ = self.mha(lstm_out, lstm_out, lstm_out)
        # print(attn_output)
        # print(attn_output.shape)
        x = attn_output.squeeze(1)
        # print(x)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)

        return self.fc4(x)




# In[211]:


# def train(model, iterator, optimizer, criterion):
#     """ Trains the model on the given training set """
#     epoch_loss = 0
#     epoch_acc = 0
    
#     model.train() # Tells your model that you are training the model
    
#     for text, labels in iterator:
        
#         # https://discuss.pytorch.org/t/how-to-add-to-attribute-to-dataset/86468
#         text = text.to(device)
#         labels = labels.to(device)
        
#         optimizer.zero_grad() # Zero the previous gradients
        
#         logits = model(text)
#         labels = labels.type_as(logits)
        
#         loss = criterion(logits, labels)
#         acc = binary_accuracy(logits, labels)
        
#         loss.backward() # Compute gradients
        
#         optimizer.step() # Make the updates
        
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
        
#     return epoch_loss/len(iterator), epoch_acc/len(iterator)


# In[27]:


import torch.nn.functional as F

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # Round predictions to the closest integer (0 or 1)
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # Convert into float for division
    acc = correct.sum() / len(correct)
    return acc


# In[28]:


i = 0
for text, labels in train_loader:
    print(f"{i}text: {text}")
    print(f"{i}labels: {labels}")
    i+=1


# In[44]:


model = ModifiedModel(input_dim=36, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.to(torch.float64)

# Define the loss function and the optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary classification
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Training loop
num_epochs = 100


# In[55]:


import os
# Re-run the training loop with accuracy computation
training_losses = []
validation_losses = []
training_accuracies = []
validation_accuracies = []
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_acc = 0
    for text, labels in train_loader:
        optimizer.zero_grad()
        # text, labels = batch
        text, labels = text.to(device), labels.to(device)
        # predictions = model(text).squeeze(1)
        # print(next(model.parameters()).dtype)
        # print(text.dtype)

        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels.float())
        acc = binary_accuracy(predictions, labels.float())
        
         # optimizer.step()
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
    training_losses.append(total_loss/len(train_loader))
    training_accuracies.append(total_acc/len(train_loader))

    print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)}, Training Accuracy: {total_acc/len(train_loader)}")

   # Validation loop
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for text, labels in val_loader:
            # text, labels = batch
            text = text.to(device)
            labels = labels.to(device)
            predictions = model(text).squeeze(1)

            # print(predictions.shape)
            loss = criterion(predictions, labels.float())
            acc = binary_accuracy(predictions, labels.float())
            val_loss += loss.item()
            val_acc += acc.item()
    if( (val_acc/len(val_loader)) > 0.8) :
       # Save the model checkpoint to the specified filepath
        filename = os.path.join('./model', f"{epoch:02d}-{val_accuracy:.2f}.pth")
        torch.save(model.state_dict(), filename)
    print(f"       Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_acc/len(val_loader)}")


# In[ ]:


def countParameters(model):
    """ Counts the total number of trainiable parameters in the model """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


# In[ ]:


trainable, frozen = countParameters(model)
print(model)
print(f"The model has {trainable:,} trainable parameters and {frozen:,} frozen parameters")


# In[ ]:




