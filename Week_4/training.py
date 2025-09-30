import xml.etree.cElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#-------get data from xml-------
tree=ET.parse('data.xml') #From file
root=tree.getroot()
ns={"cwa": "urn:cwa:gov:tw:cwacommon:0.1"}
content = root.find(".//cwa:Content", ns)
if content is not None:
    values = np.array([v for v in content.text.strip().split("\n")])
else:
    print("No Content found in XML.")

#-------classification and regression--------
data=[]
data_X=[]
data_Y=[]
regg=[]
regg_X=[]
regg_Y=[]
r=21.88#21.88-25.45
d=0.03
for v in values:
    arr=v.split(',')
    c=120#120-121.98
    for k in arr:
        indexE=np.strings.find(k, 'E')
        temp=k[:indexE]
        if(temp=='-999.0'):
            flag=0            
        else:
            flag=1            
            regg.append([float(r), float(c), float(temp)])
        if(temp!='-999.0' and (float(temp)<50.0 and float(temp)>-5.0)):
            regg_X.append([float(r), float(c)])#regg---------
            regg_Y.append(float(temp))#regg---------
        each=[float(r), float(c), flag]
        data.append(each)
        data_X.append([float(r), float(c)])
        data_Y.append(flag)
        c=round(c+d, 2)
    r=round(r+d, 2)
    
data_X=np.array(data_X)
data_Y=np.array(data_Y)
regg_X=np.array(regg_X)#regg---------
regg_Y=np.array(regg_Y)#regg---------

#-------start model training-------
class ModelC(nn.Module):
    def __init__(self,hidden_sizes=[64, 128, 128, 64]):
        super().__init__()
        layers=[]
        in_dim=2
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim=h
        layers.append(nn.Linear(in_dim, 1))
        self.net=nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ModelR(nn.Module):#regg---------
    def __init__(self,hidden_sizes=[128 for i in range(16)]):
        super().__init__()
        layers=[]
        in_dim=2
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim=h
        layers.append(nn.Linear(in_dim, 1))
        self.net=nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
    
#-----normalize
mean_temp = np.mean(regg_Y)
std_temp = np.std(regg_Y)
regg_Y_norm = (regg_Y - mean_temp) / std_temp

#-----Split into train and test
rc_train, rc_val, f_train, f_val=train_test_split(data_X, data_Y, test_size=0.2, random_state=1)
rc2_train, rc2_val, t_train, t_val=train_test_split(regg_X, regg_Y_norm, test_size=0.2, random_state=1)#regg---------
#-----Convert to torch tensors
RC_train=torch.from_numpy(rc_train).to(torch.float32)
RC_val=torch.from_numpy(rc_val).to(torch.float32)
F_train=torch.from_numpy(f_train).to(torch.float32)
F_val=torch.from_numpy(f_val).to(torch.float32)

RC2_train=torch.from_numpy(rc2_train).to(torch.float32)#regg---------
RC2_val=torch.from_numpy(rc2_val).to(torch.float32)
T_train=torch.from_numpy(t_train).to(torch.float32)
T_val=torch.from_numpy(t_val).to(torch.float32)

#-----Lr, Model, loss, optimizer
lr=1e-4
epochs=8000
model=ModelC()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=lr)
data_loss=[]

model2=ModelR()#regg---------
criterion2=nn.MSELoss()
optimizer2=optim.Adam(model2.parameters(), lr=lr)
regg_loss=[]

#-----Training loop
for v in range(epochs):
    model.train()
    optimizer.zero_grad()
    data_out=model(RC_train)
    loss=criterion(data_out, F_train.unsqueeze(1))#don't use tensor type
    loss.backward()
    data_loss.append(loss.item())
    optimizer.step()

#-----Prediction on all points
    model.eval()
    with torch.no_grad():
        data_pred=model(RC_val.to(torch.float32))
        loss_val=criterion(data_pred, F_val.unsqueeze(1))
    
    if (v+1) % 1000 == 0:
        print(f"Epoch {v+1}/{epochs}, Loss: {loss_val.item():.4f}")

#regg---------
for v in range(epochs):
    model2.train()#regg---------
    optimizer2.zero_grad()
    regg_out=model2(RC2_train.to(torch.float32))
    loss2=criterion2(regg_out, T_train.unsqueeze(1))
    loss2.backward()
    regg_loss.append(loss2.item())
    optimizer2.step()

    model2.eval()#regg---------
    with torch.no_grad():
        regg_pred=model2(RC2_val.to(torch.float32))
        loss2_val=criterion2(regg_pred, T_val.unsqueeze(1))

    if (v+1) % 1000 == 0:
        print(f"Epoch {v+1}/{epochs}, Loss: {loss2_val.item():.4f}")#regg---------

#-----plot
plt.plot(data_loss, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

plt.plot(regg_loss, label="Training Loss")#regg
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

#-----scattor
with torch.no_grad():
    outputs_all = model(torch.tensor(data_X, dtype=torch.float32))
    _, predicted_all = torch.max(outputs_all, 1)

predicted_all = predicted_all.cpu().numpy()  # 轉成 numpy array
true_labels = data_Y  # 原始標籤 (0/1)

# 真實標籤
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(data_X[:,1], data_X[:,0], c=true_labels, cmap="coolwarm", s=10)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("True Labels")
plt.colorbar(label="Label")

# 模型預測
plt.subplot(1,2,2)
plt.scatter(data_X[:,1], data_X[:,0], c=outputs_all, cmap="coolwarm", s=10)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Predicted Labels")
plt.colorbar(label="Label")

plt.tight_layout()
plt.show()


#regg-----scattor
with torch.no_grad():
    outputs_all = model2(torch.tensor(regg_X, dtype=torch.float32))
    pred = outputs_all * std_temp + mean_temp

predicted_all = pred.cpu().numpy()  # 轉成 numpy array
true_labels = regg_Y  # 原始標籤 (0/1)
# 真實標籤
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(regg_X[:,1], regg_X[:,0], c=true_labels, cmap="coolwarm", s=10)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("True Labels")
plt.colorbar(label="Label")

# 模型預測
plt.subplot(1,2,2)
plt.scatter(regg_X[:,1], regg_X[:,0], c=predicted_all, cmap="coolwarm", s=10)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Predicted Labels")
plt.colorbar(label="Label")

plt.tight_layout()
plt.show()