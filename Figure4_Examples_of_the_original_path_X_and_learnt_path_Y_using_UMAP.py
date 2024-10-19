import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import umap

def generate_data(num_points=500, noise_level=0.2):
    x = np.linspace(0, 10, num_points)
    y = np.sin(x) + noise_level * np.random.normal(size=num_points)
    return x, y

class AdvancedPathModel(nn.Module):
    def __init__(self):
        super(AdvancedPathModel, self).__init__()
        self.fc1 = nn.Linear(1, 256) 
        self.dropout = nn.Dropout(0.3) 
        self.fc2 = nn.Linear(256, 128)  
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)    
        self.fc5 = nn.Linear(32, 1)     

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

x, y = generate_data()
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

x_norm = (x - x_min) / (x_max - x_min)
y_norm = (y - y_min) / (y_max - y_min)

x_tensor = torch.tensor(x_norm, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1)

x_train, x_val, y_train, y_val = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)

model = AdvancedPathModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5) 

num_epochs = 20000  
early_stopping_patience = 500 
best_loss = float('inf')
stopping_counter = 0

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val)
        val_loss = criterion(val_outputs, y_val)

    if (epoch + 1) % 1000 == 0:
        model.eval()
        with torch.no_grad():
            predicted = model(x_tensor).numpy()

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

        embedding_original = reducer.fit_transform(np.vstack((x_norm, y_norm)).T)
        embedding_predicted = reducer.fit_transform(np.vstack((x_norm, predicted.squeeze())).T)

        plt.figure(figsize=(10, 5))
        plt.scatter(embedding_original[:, 0], embedding_original[:, 1], c='red', label='Original Path', alpha=0.5)
        plt.scatter(embedding_predicted[:, 0], embedding_predicted[:, 1], c='blue', label='Learnt Path', alpha=0.5)
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title(f'UMAP Projection of Original and Learnt Paths at Epoch {epoch+1}')
        plt.legend()
        plt.grid()
        plt.show()

    if val_loss < best_loss:
        best_loss = val_loss
        stopping_counter = 0
    else:
        stopping_counter += 1

    if stopping_counter >= early_stopping_patience:
        print("Early stopping triggered")
        break