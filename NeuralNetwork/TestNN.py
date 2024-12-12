import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
        
def load_entire_model(filepath,input_size):
    model = Net(input_size)
    model.load_state_dict(torch.load(filepath, map_location=device))  # Load the entire model
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    

    dataset = '../Datasets/Small_datasetPreprocessed.parquet'

    if os.path.exists(dataset):
        df = pd.read_parquet(dataset)
        df = df.dropna() 
        print(df.head(1))
        print(df.shape)
        #remove total_amount > 350
        df = df[df['total_amount'] <= 350]
        print(df.shape)
    else:
        print("Dataset not found")

    # Preprocess the data
    input_size = df.shape[1] - 1
    X = df.drop(['total_amount'], axis=1).values
    y = df['total_amount'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_test.shape)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)

    test_dataset = NYCTaxiDataset(X_test, y_test)

    batch_size = 512
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # Function to load the entire model
    
    # Evaluation function for regression with R² score, MAE, and MSE
    def evaluate_model(model, test_loader):
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc=f'Testing', unit='batch') as pbar:
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(inputs)
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(labels.cpu().numpy())
                    pbar.update(1)

        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)  # Calculate MAE
        r2 = r2_score(actuals, predictions)  # Calculate R² score
        
        print(f"Test MSE: {mse:.2f}")
        print(f"Test RMSE: {mse ** 0.5:.2f}")
        print(f"Test MAE: {mae:.2f}")
        print(f"Test R² Score: {r2:.4f}")

        #put in a plot the difference between the actual and predicted values
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        diff = [actuals[i] - predictions[i] for i in range(len(actuals))]
        plt.plot(diff, label='Difference between actual and predicted values')
        plt.legend()
        plt.title('Difference between actual and predicted values in the Neural Network')
        # Save the plot 
        plt.savefig('nn_plot.png')

        #PLOT THE SCORES (MAE,MSE AND R2) IN HISTOGRAMS
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(10, 5))
        scores = [mse, mae, r2]
        labels = ['MSE', 'MAE', 'R²']
        x = np.arange(len(labels))
        #put different colors to the bars and put values above the bar
        for i in range(len(scores)):
            plt.text(x[i], scores[i], str(round(scores[i], 4)), ha='center')
        colors = ['#16537e', '#cc0010', '#f1c232']
        plt.bar(x, scores, color=colors)
        plt.xticks(x, labels)
        plt.title('Neural Network Scores')
        plt.savefig('nn_scores.png')


    # Load the entire model and evaluate it
    model_filepath = 'model_LargeNN.pth'  # Replace with your .pth file path
    loaded_model = load_entire_model(model_filepath,input_size)

    # Assume test_loader is defined as before
    evaluate_model(loaded_model, test_loader)
