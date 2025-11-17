import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import numpy as np


class TemporalAttentionLayer(nn.Module):
    """Attention mechanism for temporal sequences"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        attention_weights = F.softmax(self.attention(x), dim=1)
        attended = torch.sum(attention_weights * x, dim=1)
        return attended, attention_weights


class DynamicGraphBuilder(nn.Module):
    """Builds dynamic edges based on wind patterns"""
    def __init__(self, max_distance_km=300):
        super().__init__()
        self.max_distance = max_distance_km
        
    def build_dynamic_edges(self, positions, wind_speed, wind_direction, threshold=0.3):
        """
        Args:
            positions: [N, 2] (lat, lon)
            wind_speed: [N]
            wind_direction: [N] (degrees)
            threshold: minimum influence score to create edge
        Returns:
            edge_index, edge_attr
        """
        N = positions.shape[0]
        edges = []
        edge_weights = []
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                # Calculate distance
                dist = self._haversine_distance(positions[i], positions[j])
                if dist > self.max_distance:
                    continue
                
                # Calculate directional influence
                angle_to_j = self._calculate_bearing(positions[i], positions[j])
                wind_dir_i = wind_direction[i].item()
                
                # How aligned is the wind direction with the direction to city j?
                angle_diff = abs(self._angle_difference(wind_dir_i, angle_to_j))
                directional_alignment = np.cos(np.radians(angle_diff))
                
                # Wind strength factor
                wind_factor = wind_speed[i].item() / 10.0  # Normalize to 0-1 range
                
                # Distance decay
                distance_factor = np.exp(-dist / 100.0)
                
                # Combined influence score
                influence = directional_alignment * wind_factor * distance_factor
                
                if influence > threshold:
                    edges.append([i, j])
                    edge_weights.append(influence)
        
        if len(edges) == 0:
            # Fallback: connect each node to itself
            edges = [[i, i] for i in range(N)]
            edge_weights = [1.0] * N
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        
        return edge_index, edge_attr
    
    def _haversine_distance(self, pos1, pos2):
        """Calculate distance in km between two lat/lon points"""
        lat1, lon1 = pos1[0].item(), pos1[1].item()
        lat2, lon2 = pos2[0].item(), pos2[1].item()
        
        R = 6371  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _calculate_bearing(self, pos1, pos2):
        """Calculate bearing (direction) from pos1 to pos2"""
        lat1, lon1 = np.radians(pos1[0].item()), np.radians(pos1[1].item())
        lat2, lon2 = np.radians(pos2[0].item()), np.radians(pos2[1].item())
        
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        
        bearing = np.degrees(np.arctan2(x, y))
        return (bearing + 360) % 360
    
    def _angle_difference(self, angle1, angle2):
        """Calculate smallest difference between two angles"""
        diff = (angle2 - angle1 + 180) % 360 - 180
        return diff


class SpatioTemporalHazeGNN(nn.Module):
    """
    Advanced Spatio-Temporal Graph Neural Network for Haze Prediction
    Incorporates:
    - Dynamic graph construction based on wind patterns
    - Temporal LSTM for time-series modeling
    - Graph Attention for spatial dependencies
    - Multi-head attention for feature importance
    """
    def __init__(self, node_features, edge_features, hidden_dim=64, 
                 num_heads=4, lstm_layers=2, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dynamic_graph_builder = DynamicGraphBuilder()
        
        # Initial feature transformation
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph Attention Layers for spatial modeling
        self.gat1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                            dropout=dropout, edge_dim=edge_features)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, 
                            dropout=dropout, edge_dim=edge_features)
        
        # Temporal LSTM for time-series patterns
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttentionLayer(hidden_dim)
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensures positive uncertainty
        )
        
    def forward(self, data, return_attention=False, return_uncertainty=False):
        """
        Args:
            data: PyG Data object with:
                - x: node features [N, node_features]
                - edge_index: static edges (can be None if using dynamic)
                - edge_attr: edge features [E, edge_features]
                - positions: [N, 2] lat/lon
                - wind_speed: [N]
                - wind_direction: [N]
                - temporal_features: [N, seq_len, temporal_dim] (optional)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Build dynamic graph based on current wind patterns
        if hasattr(data, 'positions') and hasattr(data, 'wind_speed'):
            edge_index, edge_attr = self.dynamic_graph_builder.build_dynamic_edges(
                data.positions, data.wind_speed, data.wind_direction
            )
        
        # Encode node features
        x = self.node_encoder(x)
        
        # Spatial graph convolutions with attention
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        spatial_features = self.gat2(x, edge_index, edge_attr)
        
        # Temporal modeling (if temporal features available)
        if hasattr(data, 'temporal_features') and data.temporal_features is not None:
            temporal_x = data.temporal_features
            lstm_out, _ = self.lstm(temporal_x)
            temporal_features, temporal_attn = self.temporal_attention(lstm_out)
        else:
            # If no temporal features, use spatial features
            temporal_features = spatial_features
            temporal_attn = None
        
        # Combine spatial and temporal
        combined_features = torch.cat([spatial_features, temporal_features], dim=1)
        
        # Prediction
        prediction = self.predictor(combined_features)
        
        output = {'prediction': prediction}
        
        if return_uncertainty:
            uncertainty = self.uncertainty_head(combined_features)
            output['uncertainty'] = uncertainty
        
        if return_attention and temporal_attn is not None:
            output['temporal_attention'] = temporal_attn
            
        return output


class HazeHorizonSimulator(nn.Module):
    """
    What-if scenario simulator for analyst-drawn fire zones
    Uses physics-informed neural network approach
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Physics-based dispersion parameters (learnable)
        self.dispersion_rate = nn.Parameter(torch.tensor(0.5))
        self.wind_transfer_coef = nn.Parameter(torch.tensor(0.7))
        
    def simulate_new_fire(self, data, new_fire_location, fire_intensity, simulation_hours=24):
        """
        Simulate haze from a new hypothetical fire
        
        Args:
            data: Current system state
            new_fire_location: [lat, lon]
            fire_intensity: scalar (0-100)
            simulation_hours: how many hours to simulate
        """
        predictions = []
        current_data = data.clone()
        
        # Add fire influence to nearby nodes
        distances = torch.norm(
            current_data.positions - new_fire_location.unsqueeze(0), 
            dim=1
        )
        
        # Gaussian influence based on distance
        fire_influence = fire_intensity * torch.exp(-distances / 50.0)
        
        for hour in range(simulation_hours):
            # Update features with fire influence
            current_data.x[:, 5] += fire_influence * (0.9 ** hour)  # Decay over time
            
            # Get prediction
            output = self.base_model(current_data, return_uncertainty=True)
            predictions.append(output)
            
            # Update state for next iteration (simplified)
            # In reality, would use weather forecast updates
            
        return predictions


def create_training_masks(num_nodes, train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test masks for temporal split"""
    indices = torch.randperm(num_nodes)
    
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    return train_mask, val_mask, test_mask
