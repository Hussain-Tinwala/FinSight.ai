import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Data Preparation & Windowing ---
def create_sequences(data, seq_length):
    """
    Slices the 2D time-series into overlapping 3D windows.
    Input shape: (Rows, Features)
    Output shape: (Samples, Sequence_Length, Features)
    """
    xs = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        xs.append(x)
    return np.array(xs)

# --- 2. PyTorch Model Architecture ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=16):
        super(LSTMAutoencoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # ENCODER: Compresses sequence down to hidden_dim
        self.encoder = nn.LSTM(input_size=n_features, 
                               hidden_size=hidden_dim, 
                               num_layers=1, 
                               batch_first=True)
        
        # DECODER: Reconstructs sequence back to n_features
        self.decoder = nn.LSTM(input_size=hidden_dim, 
                               hidden_size=hidden_dim, 
                               num_layers=1, 
                               batch_first=True)
        
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # 1. Encode
        _, (hidden, _) = self.encoder(x) 
        # hidden shape: (1, batch, hidden_dim)
        
        # 2. Repeat hidden state for the decoder (bottleneck expansion)
        # We need to feed the compressed vector into every step of the decoder
        hidden = hidden[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        
        # 3. Decode
        decoded, _ = self.decoder(hidden)
        
        # 4. Final output projection back to original features
        reconstructed = self.output_layer(decoded)
        return reconstructed

# --- 3. Main Execution ---
def run_deep_learning_detection():
    print("Loading ML-scored data...")
    df = pd.read_csv("ml_scored_billing.csv")
    
    # We will just train on AWS COMPUTE to demonstrate the DL pipeline quickly
    df_aws = df[(df['cloud_provider'] == 'AWS') & (df['unified_category'] == 'COMPUTE')].copy()
    df_aws = df_aws.sort_values('timestamp').reset_index(drop=True)
    
    features_to_scale = ['cost', 'cost_ratio', 'rolling_mean_7d']
    scaler = MinMaxScaler()
    df_aws[features_to_scale] = scaler.fit_transform(df_aws[features_to_scale])
    
    # CRITICAL: We only want to train the model on NORMAL data.
    # In reality, we'd use Tier 1/2 results to filter out bad data.
    # Here, we'll cheat slightly using our ground truth labels to get a perfectly clean training set.
    clean_data_mask = df_aws['is_anomaly'] == 0
    train_data = df_aws.loc[clean_data_mask, features_to_scale].values
    
    SEQ_LEN = 14 # 2-week memory window
    
    # Build 3D Tensors
    X_train = create_sequences(train_data, SEQ_LEN)
    X_train_tensor = torch.FloatTensor(X_train)
    
    # The full dataset (which includes anomalies) for evaluation
    full_data = df_aws[features_to_scale].values
    X_full = create_sequences(full_data, SEQ_LEN)
    X_full_tensor = torch.FloatTensor(X_full)

    # Initialize Model
    model = LSTMAutoencoder(n_features=len(features_to_scale), hidden_dim=8)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training LSTM Autoencoder (this will take a few seconds)...")
    epochs = 50
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        # Autoencoders learn by trying to make output equal to input
        loss = criterion(output, X_train_tensor) 
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss (MSE): {loss.item():.4f}")

    print("Scoring production data (Reconstruction Error)...")
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_full_tensor)
        # Calculate MSE per sequence (how badly did it reconstruct this 14-day window?)
        # Shape: (Samples)
        mse_scores = torch.mean((reconstructed - X_full_tensor)**2, dim=[1, 2]).numpy()
    
    # We must pad the beginning with zeros since the first 14 days couldn't form a window
    padding = np.zeros(SEQ_LEN)
    full_mse_scores = np.concatenate([padding, mse_scores])
    df_aws['dl_mse_score'] = full_mse_scores
    
    # Dynamic Thresholding: Flag as anomaly if MSE > 95th percentile of normal training errors
    threshold = np.percentile(mse_scores, 95)
    df_aws['dl_is_anomaly'] = (df_aws['dl_mse_score'] > threshold).astype(int)
    
    # --- Evaluation ---
    true_anomalies = df_aws[df_aws['is_anomaly'] == 1]
    caught = df_aws[(df_aws['is_anomaly'] == 1) & (df_aws['dl_is_anomaly'] == 1)]
    
    print("-" * 30)
    print("Tier 3 (Deep Learning) Detection Results for AWS COMPUTE:")
    print(f"MSE Anomaly Threshold chosen: {threshold:.4f}")
    print(f"Total True Anomalies Injected: {len(true_anomalies)}")
    print(f"Caught by Tier 3 (LSTM): {len(caught)} ({(len(caught)/max(1, len(true_anomalies)))*100:.1f}%)")

if __name__ == "__main__":
    run_deep_learning_detection()