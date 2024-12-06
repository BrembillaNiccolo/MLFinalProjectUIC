import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# Check the data in the 2 created datasets (2019newBig.csv: 12M rows, 2019new.csv: 1.2M rows)

# %% [markdown]
# Load the DB and cleanup

# %%

# %% [markdown]
# Start models training with different NN and parameter to see the best ones



class Net(nn.Module):
    def __init__(self,input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)  # Larger layer with 512 neurons
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)         # Another large layer with 256 neurons            
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)         # 128 neurons
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)          # 64 neurons
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)           # 32 neurons
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, 1)            # Output layer

        self.dropout = nn.Dropout(0.4)         # Increased dropout rate to combat overfitting

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        return x


    # Custom dataset class
class NYCTaxiDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
# %%

if __name__ == '__main__':
    dataset = '../Datasets/Small_datasetPreprocessed.parquet'

    if os.path.exists(dataset):
        df = pd.read_parquet(dataset)
        df = df.dropna() 
        print(df.head(1))
        print(df.shape)
    else:
        print("Dataset not found")

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)


    # Preprocess the data
    input_size = df.shape[1] - 1
    X = df.drop(['total_amount'], axis=1).values
    y = df['total_amount'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)

    train_dataset = NYCTaxiDataset(X_train, y_train)
    test_dataset = NYCTaxiDataset(X_test, y_test)

    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Initialize the model and wrap it with DataParallel if multiple GPUs are available
    model = Net(input_size)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training loop with TF32 enabled
    def train_model(model, train_loader, num_epochs=50):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            with tqdm(total=len(train_loader), desc=f'Epoch [{epoch+1}/{num_epochs}]', unit='batch') as pbar:
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    pbar.update(1)

            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            scheduler.step(epoch_loss)
            if(epoch%10==0):
                evaluate_model(model, test_loader)

    # Evaluation function for regression
    def evaluate_model(model, test_loader):
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(labels.cpu().numpy())

        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)  # Calculate MAE
        r2 = r2_score(actuals, predictions)  # Calculate R² score
        
        print(f"Test MSE: {mse:.2f}")
        print(f"Test RMSE: {mse ** 0.5:.2f}")
        print(f"Test MAE: {mae:.2f}")
        print(f"Test R² Score: {r2:.4f}")

    # Train the model
    train_model(model, train_loader)

    # Evaluate the model
    evaluate_model(model, test_loader)

    # Save the trained model
    torch.save(model.state_dict(), 'model_LargeNN.pth')

    print("Training and evaluation completed!")
