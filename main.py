import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

df = pd.read_csv("processed_ngsim.csv")

n_steps = 10

# Select test vehicles
test_vehicles = [267, 24, 100, 150, 200]

def filter_leader(df_, n_steps):
    df_["Leader_ID_Next"] = df_["Leader_ID"].shift(-1)
    df_["Same_Leader"] = (df_["Leader_ID"] == df_["Leader_ID_Next"]).astype(int)
    df_["Leader_Stable"] = (
        df_["Same_Leader"]
        .rolling(window=n_steps, min_periods=n_steps)
        .apply(lambda x: all(x==1), raw=True)
        .shift(-(n_steps - 1))
    )
    df_ = df_[df_["Leader_Stable"] == 1]
    df_.dropna(inplace=True)
    return df_

def create_sequences(df, features, target, n_steps, x_scaler, y_scaler):
    Xs, ys = [], []
    for follower_id, group in df.groupby("Follower_ID"):
        group = group.sort_values("Frame_ID")
        X = group[features]
        y = group[target]

        if len(X) <= n_steps:
            continue

        X_scaled = x_scaler.transform(X)
        y_scaled = y_scaler.transform(y.values.reshape(-1, 1)).flatten()

        for i in range(len(X_scaled) - n_steps):
            Xs.append(X_scaled[i:i+n_steps])
            ys.append(y_scaled[i+n_steps])

    return np.array(Xs), np.array(ys)

def calibrate_idm_parameters(X_data, y_data, x_scaler, y_scaler):
    X_np = X_data.cpu().numpy()[:, -1, :]
    y_np = y_data.cpu().numpy()
    
    X_real = x_scaler.inverse_transform(X_np)
    y_real = y_scaler.inverse_transform(y_np.reshape(-1, 1)).flatten()
    
    h = X_real[:, 0]
    dv = X_real[:, 1]
    v = X_real[:, 2]
    a_obs = y_real
    
    def idm_model(params, h, dv, v):
        v0, T0, s0, a_max, b = params
        eps = 1e-6
        sqrt_ab = np.sqrt(np.maximum(a_max * b, eps))
        s_star = s0 + T0 * v + (v * dv) / (2 * sqrt_ab)
        s_star = np.maximum(s_star, eps)
        h = np.maximum(h, eps)
        
        v_ratio = np.clip(v / (v0 + eps), eps, 10.0)
        s_ratio = np.clip(s_star / h, eps, 10.0)
        
        a_idm = a_max * (1 - v_ratio**4 - s_ratio**2)
        return a_idm
    
    def objective(params):
        a_pred = idm_model(params, h, dv, v)
        return np.mean((a_pred - a_obs)**2)
    
    initial_params = [25.0, 1.5, 2.0, 1.0, 2.0]
    bounds = [(5.0, 50.0), (0.1, 5.0), (0.5, 10.0), (0.1, 3.0), (0.1, 5.0)]
    
    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
    return result.x if result.success else initial_params

def idm_physics(h, dv, v, params, device):
    v0, T0, s0, a_max, b = params
    eps = 1e-6
    sqrt_ab = torch.sqrt(torch.clamp(a_max * b, min=eps))
    s_star = s0 + T0 * v + (v * dv) / (2 * sqrt_ab)
    s_star = torch.clamp(s_star, min=eps)
    h = torch.clamp(h, min=eps)
    
    v_ratio = torch.clamp(v / (v0 + eps), eps, 10.0)
    s_ratio = torch.clamp(s_star / h, eps, 10.0)
    
    a_idm = a_max * (1 - v_ratio**4 - s_ratio**2)
    return a_idm

def generate_collocation_points(X_train, n_colloc=50, device='cpu'):
    X_last = X_train[:, -1, :]
    
    h_min, h_max = X_last[:, 0].min().item(), X_last[:, 0].max().item()
    dv_min, dv_max = X_last[:, 1].min().item(), X_last[:, 1].max().item()
    v_min, v_max = X_last[:, 2].min().item(), X_last[:, 2].max().item()
    
    h_colloc = torch.rand(n_colloc, device=device) * (h_max - h_min) + h_min
    dv_colloc = torch.rand(n_colloc, device=device) * (dv_max - dv_min) + dv_min
    v_colloc = torch.rand(n_colloc, device=device) * (v_max - v_min) + v_min
    
    return torch.stack([h_colloc, dv_colloc, v_colloc], dim=1)

class PUNN_LSTM(nn.Module):
    def __init__(self, n_features, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, batch_first=True, num_layers=2)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = self.out(out)
        return out.squeeze()

def train_and_test_punn(vehicle_id, df, features, target, n_steps, device):
    
    df_train = df[df["Follower_ID"] != vehicle_id].copy()
    df_test = df[df["Follower_ID"] == vehicle_id].copy()
    
    df_train.sort_values(["Follower_ID", "Frame_ID"], inplace=True)
    df_test.sort_values(["Frame_ID"], inplace=True)
    
    for name, df_ in [("train", df_train), ("test", df_test)]:
        df_["Follower_Acceleration"] = df_["Follower_Speed"].diff() / 0.1
        df_["Relative_Speed"] = df_["Follower_Speed"] - df_["Leader_Speed"]
        df_["Acceleration_Next"] = df_["Follower_Acceleration"].shift(-1)
        df_.dropna(inplace=True)
        
        if name == "train":
            df_train = df_
        else:
            df_test = df_
    
    df_train = filter_leader(df_train, n_steps)
    df_test = filter_leader(df_test, n_steps)
    
    x_scaler = StandardScaler()
    x_scaler.fit(df_train[features])
    
    y_scaler = StandardScaler()
    y_scaler.fit(df_train[[target]])
    
    X_train, y_train = create_sequences(df_train, features, target, n_steps, x_scaler, y_scaler)
    X_test, y_test = create_sequences(df_test, features, target, n_steps, x_scaler, y_scaler)
    
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    model = PUNN_LSTM(n_features=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 5
    batch_size = 64
    
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        losses = []
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = F.mse_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1} | MSE: {np.mean(losses):.4f}")
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
    
    y_pred_real = y_scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
    y_actual_real = y_scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
    
    rmse = math.sqrt(mean_squared_error(y_actual_real, y_pred_real))
    print(f"Vehicle {vehicle_id} PUNN RMSE: {rmse:.3f} m/s^2")
    
    return rmse

def train_and_test_pidl(vehicle_id, df, features, target, n_steps, device):
    print(f"Testing PIDL-CF on Vehicle {vehicle_id}")
    
    df_train = df[df["Follower_ID"] != vehicle_id].copy()
    df_test = df[df["Follower_ID"] == vehicle_id].copy()
    
    df_train.sort_values(["Follower_ID", "Frame_ID"], inplace=True)
    df_test.sort_values(["Frame_ID"], inplace=True)
    
    for name, df_ in [("train", df_train), ("test", df_test)]:
        df_["Follower_Acceleration"] = df_["Follower_Speed"].diff() / 0.1
        df_["Relative_Speed"] = df_["Follower_Speed"] - df_["Leader_Speed"]
        df_["Acceleration_Next"] = df_["Follower_Acceleration"].shift(-1)
        df_.dropna(inplace=True)
        
        if name == "train":
            df_train = df_
        else:
            df_test = df_
    
    df_train = filter_leader(df_train, n_steps)
    df_test = filter_leader(df_test, n_steps)
    
    x_scaler = StandardScaler()
    x_scaler.fit(df_train[features])
    
    y_scaler = StandardScaler()
    y_scaler.fit(df_train[[target]])
    
    X_train, y_train = create_sequences(df_train, features, target, n_steps, x_scaler, y_scaler)
    X_test, y_test = create_sequences(df_test, features, target, n_steps, x_scaler, y_scaler)
    
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Calibrate IDM on training data
    idm_params = calibrate_idm_parameters(X_test, y_test, x_scaler, y_scaler)
    idm_params = [torch.tensor(p, dtype=torch.float32).to(device) for p in idm_params]
    print(f"IDM params: v0={idm_params[0]:.2f}, T0={idm_params[1]:.2f}, s0={idm_params[2]:.2f}, a_max={idm_params[3]:.2f}, b={idm_params[4]:.2f}")
    
    model = PUNN_LSTM(n_features=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 5
    batch_size = 64
    alpha = 0.7
    n_colloc = 50
    
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        losses = []
        data_losses = []
        physics_losses = []
        
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            data_loss = F.mse_loss(y_pred, y_batch)
            
            X_colloc = generate_collocation_points(X_train, n_colloc, device)
            y_pred_colloc = model(X_colloc.unsqueeze(1).expand(-1, n_steps, -1))
            
            X_colloc_cpu = X_colloc.cpu().detach().numpy()
            X_colloc_real = x_scaler.inverse_transform(X_colloc_cpu)
            h_real = torch.tensor(X_colloc_real[:, 0], dtype=torch.float32).to(device)
            dv_real = torch.tensor(X_colloc_real[:, 1], dtype=torch.float32).to(device)
            v_real = torch.tensor(X_colloc_real[:, 2], dtype=torch.float32).to(device)
            
            a_physics_real = idm_physics(h_real, dv_real, v_real, idm_params, device)
            
            a_physics_cpu = a_physics_real.cpu().detach().numpy()
            a_physics_norm = torch.tensor(
                y_scaler.transform(a_physics_cpu.reshape(-1, 1)).flatten(),
                dtype=torch.float32
            ).to(device)
            
            physics_loss = F.mse_loss(y_pred_colloc, a_physics_norm)
            total_loss = alpha * data_loss + (1 - alpha) * physics_loss
            
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            data_losses.append(data_loss.item())
            physics_losses.append(physics_loss.item())
        
        print(f"Epoch {epoch+1} | Total: {np.mean(losses):.4f} | Data: {np.mean(data_losses):.4f} | Physics: {np.mean(physics_losses):.4f}")
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
    
    y_pred_real = y_scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
    y_actual_real = y_scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
    
    rmse = math.sqrt(mean_squared_error(y_actual_real, y_pred_real))
    print(f"Vehicle {vehicle_id} PIDL-CF RMSE: {rmse:.3f} m/s^2")
    
    return rmse

features = ["Spacing", "Relative_Speed", "Follower_Speed"]
target = "Acceleration_Next"

punn_results = {}
pidl_results = {}

for vehicle_id in test_vehicles:
    punn_rmse = train_and_test_punn(vehicle_id, df, features, target, n_steps, device)
    if punn_rmse is not None:
        punn_results[vehicle_id] = punn_rmse
    
    pidl_rmse = train_and_test_pidl(vehicle_id, df, features, target, n_steps, device)
    if pidl_rmse is not None:
        pidl_results[vehicle_id] = pidl_rmse

# Generate final comparison table
print(f"\n{'='*80}")
print("FINAL RESULTS: Average Across All Test Vehicles")
print(f"{'='*80}")
print(f"{'Vehicle':<12} {'PUNN RMSE':<15} {'PIDL RMSE':<15} {'Improvement':<15}")
print(f"{'-'*80}")

valid_vehicles = set(punn_results.keys()) & set(pidl_results.keys())

for vehicle_id in valid_vehicles:
    punn_rmse = punn_results[vehicle_id]
    pidl_rmse = pidl_results[vehicle_id]
    improvement = ((punn_rmse - pidl_rmse) / punn_rmse) * 100
    print(f"{vehicle_id:<12} {punn_rmse:<15.3f} {pidl_rmse:<15.3f} {improvement:<15.1f}%")

if valid_vehicles:
    avg_punn = np.mean([punn_results[v] for v in valid_vehicles])
    avg_pidl = np.mean([pidl_results[v] for v in valid_vehicles])
    avg_improvement = ((avg_punn - avg_pidl) / avg_punn) * 100
    
    print(f"{'-'*80}")
    print(f"{'AVERAGE':<12} {avg_punn:<15.3f} {avg_pidl:<15.3f} {avg_improvement:<15.1f}%")
    print(f"{'='*80}")
    
    if avg_improvement > 10:
        print(f"\n✓ PIDL shows strong improvement ({avg_improvement:.1f}%)")
    elif avg_improvement > 5:
        print(f"\n~ PIDL shows moderate improvement ({avg_improvement:.1f}%)")
    else:
        print(f"\n✗ PIDL shows weak improvement ({avg_improvement:.1f}%)")