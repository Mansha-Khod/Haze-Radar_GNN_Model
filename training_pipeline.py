import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineering:
    """Advanced feature engineering for haze prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        
    def create_temporal_features(self, df):
        """Create time-based features"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Cyclical encoding for hour and day
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Day of week (weekends might have different patterns)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def create_spatial_features(self, df):
        """Create location-based features"""
        df = df.copy()
        
        # Wind components
        df['wind_u'] = df['wind_speed'] * np.sin(np.radians(df['wind_direction']))
        df['wind_v'] = df['wind_speed'] * np.cos(np.radians(df['wind_direction']))
        
        # Distance from equator (affects weather patterns)
        df['dist_from_equator'] = np.abs(df['latitude'])
        
        return df
    
    def create_fire_features(self, df):
        """Enhanced fire-related features"""
        df = df.copy()
        
        # Weighted fire intensity
        df['fire_intensity'] = df['upwind_fire_count'] * (df['avg_fire_confidence'] / 100.0)
        
        # Fire risk categories
        df['high_fire_risk'] = ((df['upwind_fire_count'] > 5) & 
                                (df['avg_fire_confidence'] > 60)).astype(int)
        
        # Interaction between fires and wind
        df['fire_wind_interaction'] = df['fire_intensity'] * df['wind_speed']
        
        return df
    
    def create_atmospheric_features(self, df):
        """Weather-based derived features"""
        df = df.copy()
        
        # Heat index (simplified)
        df['heat_index'] = df['temperature'] * (1 + 0.02 * df['humidity'])
        
        # Atmospheric stability indicator
        df['stability_index'] = df['temperature'] / (df['humidity'] + 1)
        
        # Ventilation coefficient (dispersal capacity)
        df['ventilation_coef'] = df['wind_speed'] * (100 - df['humidity']) / 100
        
        return df
    
    def create_lagged_features(self, df, lag_hours=[1, 3, 6, 12, 24]):
        """Create lagged features for temporal patterns"""
        df = df.copy()
        df = df.sort_values('timestamp')
        
        for lag in lag_hours:
            df[f'aqi_lag_{lag}h'] = df.groupby('city')['current_aqi'].shift(lag)
            df[f'pm25_lag_{lag}h'] = df.groupby('city')['target_pm25_24h'].shift(lag)
        
        # Fill NaN with forward fill, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def engineer_all_features(self, df, create_lags=True):
        """Apply all feature engineering"""
        df = self.create_temporal_features(df)
        df = self.create_spatial_features(df)
        df = self.create_fire_features(df)
        df = self.create_atmospheric_features(df)
        
        if create_lags:
            df = self.create_lagged_features(df)
        
        return df
    
    def fit_scaler(self, df, feature_cols):
        """Fit scaler on training data"""
        self.scaler.fit(df[feature_cols])
        self.fitted = True
        
    def transform(self, df, feature_cols):
        """Transform features"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        df_scaled = df.copy()
        df_scaled[feature_cols] = self.scaler.transform(df[feature_cols])
        return df_scaled


class HazeTrainer:
    """Complete training pipeline with validation"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.feature_engineer = FeatureEngineering()
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'train_mae': [], 'val_mae': [], 'test_mae': []
        }
        
    def prepare_data(self, df, city_graph):
        """Prepare data with proper feature engineering"""
        
        # Feature engineering
        df = self.feature_engineer.engineer_all_features(df)
        
        # Define feature columns
        base_features = [
            "temperature", "humidity", "wind_speed", "wind_direction",
            "wind_u", "wind_v",  # Wind components
            "avg_fire_confidence", "upwind_fire_count", "fire_intensity",
            "fire_wind_interaction",
            "current_aqi", "population_density",
            "heat_index", "stability_index", "ventilation_coef",
            "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend",
            "dist_from_equator"
        ]
        
        # Add lagged features if available
        lagged_features = [col for col in df.columns if 'lag' in col]
        feature_cols = base_features + lagged_features
        
        # Filter to available columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Temporal train/val/test split
        df = df.sort_values('timestamp')
        n = len(df)
        train_idx = int(0.7 * n)
        val_idx = int(0.85 * n)
        
        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]
        
        # Fit scaler on training data only
        self.feature_engineer.fit_scaler(train_df, feature_cols)
        
        # Transform all splits
        train_df = self.feature_engineer.transform(train_df, feature_cols)
        val_df = self.feature_engineer.transform(val_df, feature_cols)
        test_df = self.feature_engineer.transform(test_df, feature_cols)
        
        # Create PyG data objects
        train_data = self._create_pyg_data(train_df, city_graph, feature_cols)
        val_data = self._create_pyg_data(val_df, city_graph, feature_cols)
        test_data = self._create_pyg_data(test_df, city_graph, feature_cols)
        
        return train_data, val_data, test_data, feature_cols
    
    def _create_pyg_data(self, df, city_graph, feature_cols):
        """Create PyTorch Geometric Data object"""
        from torch_geometric.data import Data
        
        # Node features
        x = torch.tensor(df[feature_cols].values, dtype=torch.float)
        
        # Target
        y = torch.tensor(df['target_pm25_24h'].values, dtype=torch.float).view(-1, 1)
        
        # Positions (lat, lon)
        positions = torch.tensor(df[['latitude', 'longitude']].values, dtype=torch.float)
        
        # Wind data for dynamic graph
        wind_speed = torch.tensor(df['wind_speed'].values, dtype=torch.float)
        wind_direction = torch.tensor(df['wind_direction'].values, dtype=torch.float)
        
        # City mapping
        city_to_idx = {city: i for i, city in enumerate(df['city'].unique())}
        
        # Build edges from city_graph
        edges = []
        for _, row in city_graph.iterrows():
            src = city_to_idx.get(row['city'])
            if src is None:
                continue
            
            connected = str(row['connected_cities']).split(',')
            for c in connected:
                c = c.strip()
                dst = city_to_idx.get(c)
                if dst is not None:
                    edges.append([src, dst])
        
        if len(edges) == 0:
            # Fallback: self-loops
            edges = [[i, i] for i in range(len(city_to_idx))]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.ones(edge_index.shape[1], 1)  # Default edge weights
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            positions=positions,
            wind_speed=wind_speed,
            wind_direction=wind_direction
        )
        
        return data
    
    def train_epoch(self, data, optimizer, criterion):
        """Single training epoch"""
        self.model.train()
        optimizer.zero_grad()
        
        data = data.to(self.device)
        output = self.model(data, return_uncertainty=True)
        
        pred = output['prediction']
        uncertainty = output['uncertainty']
        
        # Main prediction loss
        pred_loss = criterion(pred, data.y)
        
        # Uncertainty loss (negative log likelihood)
        nll_loss = 0.5 * torch.mean(
            (pred - data.y) ** 2 / uncertainty + torch.log(uncertainty)
        )
        
        # Combined loss
        loss = pred_loss + 0.1 * nll_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, data, criterion):
        """Evaluate on validation/test set"""
        self.model.eval()
        data = data.to(self.device)
        
        output = self.model(data, return_uncertainty=True)
        pred = output['prediction']
        
        loss = criterion(pred, data.y).item()
        
        # Convert to numpy for metrics
        pred_np = pred.cpu().numpy().flatten()
        true_np = data.y.cpu().numpy().flatten()
        
        mae = mean_absolute_error(true_np, pred_np)
        rmse = np.sqrt(mean_squared_error(true_np, pred_np))
        r2 = r2_score(true_np, pred_np)
        
        return {
            'loss': loss,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def train(self, train_data, val_data, test_data, 
              epochs=200, lr=0.001, patience=20, save_path='best_model.pth'):
        """Full training loop with early stopping"""
        
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = nn.MSELoss()
        
        patience_counter = 0
        
        print(f"Training on device: {self.device}")
        print(f"Train samples: {train_data.x.shape[0]}")
        print(f"Val samples: {val_data.x.shape[0]}")
        print(f"Test samples: {test_data.x.shape[0]}")
        print("-" * 60)
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Training
            train_loss = self.train_epoch(train_data, optimizer, criterion)
            
            # Validation
            val_metrics = self.evaluate(val_data, criterion)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mae'].append(val_metrics['mae'])
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': self.best_val_loss,
                    'val_metrics': val_metrics,
                    'feature_cols': train_data.x.shape[1]
                }, save_path)
            else:
                patience_counter += 1
            
            # Logging
            if (epoch + 1) % 20 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f} | MAE: {val_metrics['mae']:.4f} | R²: {val_metrics['r2']:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model and evaluate on test set
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.evaluate(test_data, criterion)
        
        print("\n" + "="*60)
        print("FINAL TEST SET PERFORMANCE:")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test MAE: {test_metrics['mae']:.4f}")
        print(f"Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"Test R²: {test_metrics['r2']:.4f}")
        print("="*60)
        
        return test_metrics


def train_model_pipeline(gnn_train, city_graph, save_dir='models/'):
    """Complete pipeline to train and save model"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize model
    from improved_gnn_model import SpatioTemporalHazeGNN
    
    # Model will be initialized after we know the number of features
    trainer = None
    
    # Prepare data (this also does feature engineering)
    print("Preparing data with feature engineering...")
    temp_trainer = HazeTrainer(
        SpatioTemporalHazeGNN(
            node_features=10,  # Temporary, will be updated
            edge_features=1,
            hidden_dim=64,
            num_heads=4,
            lstm_layers=2,
            dropout=0.2
        )
    )
    
    train_data, val_data, test_data, feature_cols = temp_trainer.prepare_data(
        gnn_train, city_graph
    )
    
    num_features = train_data.x.shape[1]
    print(f"Number of engineered features: {num_features}")
    
    # Initialize actual model with correct dimensions
    model = SpatioTemporalHazeGNN(
        node_features=num_features,
        edge_features=1,
        hidden_dim=64,
        num_heads=4,
        lstm_layers=2,
        dropout=0.2
    )
    
    trainer = HazeTrainer(model)
    trainer.feature_engineer = temp_trainer.feature_engineer
    
    # Train
    print("\nStarting training...")
    test_metrics = trainer.train(
        train_data, val_data, test_data,
        epochs=200,
        lr=0.001,
        patience=20,
        save_path=os.path.join(save_dir, 'best_haze_model.pth')
    )
    
    # Save feature engineering config
    config = {
        'feature_cols': feature_cols,
        'num_features': num_features,
        'test_metrics': test_metrics,
        'training_date': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel and config saved to {save_dir}")
    
    return trainer, test_metrics
