# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_he(m):
    if isinstance(m, nn.Linear):
        # Kaiming Uniform initialization with 'relu' nonlinearity.
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class JointStanceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        A simple linear classifier for joint document-topic embeddings.
        
        Args:
            input_dim (int): Dimension of the input features (should be 4096).
            num_classes (int): Number of output classes.
        """
        super(JointStanceClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        # Apply Xavier initialization
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        logits = self.linear(x)
        return logits

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class JointMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.1, num_layers=4):
        """
        An improved MLP classifier with SwiGLU activations, layer normalization, and prenorm architecture.
        
        Args:
            input_dim (int): Dimension of input features (4096).
            hidden_dim (int): Dimension of the hidden layers.
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout rate.
            num_layers (int): Number of hidden layers.
        """
        super(JointMLPClassifier, self).__init__()
        
        # Input layer normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # First layer projects to 2x hidden_dim for SwiGLU
        self.input_proj = nn.Linear(input_dim, hidden_dim * 2)
        
        # Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'linear': nn.Linear(hidden_dim, hidden_dim * 2),
                'dropout': nn.Dropout(dropout_rate)
            })
            self.layers.append(layer)
            
        # Output head
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_classes)
        
        # Activation
        self.act = SwiGLU()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Using Kaiming He uniform initialization
            # The nonlinearity is 'relu' since we're using SwiGLU which uses SiLU activation
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        # Input normalization and projection
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = self.act(x)
        
        # Hidden layers with residual connections
        for layer in self.layers:
            residual = x
            x = layer['norm'](x)
            x = layer['linear'](x)
            x = self.act(x)
            x = layer['dropout'](x)
            x = x + residual
            
        # Output head
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        return logits
