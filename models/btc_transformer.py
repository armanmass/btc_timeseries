import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inputs of shape (seq_len, batch_size, d_model)"""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class BTCTransformer(nn.Module):
    """Time Series Transformer model for BTC prediction."""
    def __init__(
        self, 
        num_features: int, 
        input_window: int,
        prediction_horizons: list[int],
        d_model: int = 128, 
        nhead: int = 8, 
        num_encoder_layers: int = 3, 
        dim_feedforward: int = 512, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_type = 'Time Series Transformer'
        self.input_window = input_window
        self.prediction_horizons = prediction_horizons
        self.d_model = d_model

        # Input embedding layer: Project raw features to d_model dimensions
        self.embedding = nn.Linear(num_features, d_model)

        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(d_model, dropout, input_window)

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # Set batch_first to True for (batch_size, seq_len, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )

        # Output layers for multi-horizon prediction
        # Each horizon gets a separate linear layer
        self.output_layers = nn.ModuleList([
            nn.Linear(d_model * input_window, 1) for _ in prediction_horizons # Predict 1 value per horizon
        ])
        
        # Optional: Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor of shape (batch_size, input_window, num_features)
        
        Returns:
            Tensor of shape (batch_size, sum(prediction_horizons)) or similar,
            depending on how targets are structured.
            For simplicity, we'll output shape (batch_size, num_horizons)
        """
        # src shape: (batch_size, input_window, num_features)

        # 1. Input Embedding
        # Output shape: (batch_size, input_window, d_model)
        src = self.embedding(src)

        # 2. Positional Encoding (expects shape (seq_len, batch_size, d_model) by default, convert if batch_first=False)
        # If batch_first=True in TransformerEncoderLayer, PositionalEncoding might need adjustment or apply before unsqueeze(1)
        # Let's adjust PositionalEncoding or apply transpose if needed. Default PE expects (Seq, Batch, Feature).
        # With batch_first=True in encoder layer, input to encoder is (Batch, Seq, Feature).
        # So, apply PE *before* the transpose if PE is (Seq, 1, Feature) and input is (Seq, Batch, Feature).
        # If input is (Batch, Seq, Feature), need to transpose for default PE, or adjust PE.
        # Let's stick to batch_first=True and adjust PE application or PE module if necessary.
        # A simpler approach for batch_first=True is to ensure PE is calculated for (Seq, Feature) and broadcasting happens.
        # Let's fix PositionalEncoding to output (Seq, 1, d_model) and assume broadcasting with (Batch, Seq, d_model)
        # However, the PE forward expects (Seq, Batch, d_model), let's adjust it or transpose input.
        # Let's adjust PE application in forward pass for batch_first=True encoder.

        # Current PE expects (seq_len, batch_size, d_model). Input src is (batch_size, input_window, d_model)
        # Transpose src for PE:
        src = src.permute(1, 0, 2) # Shape: (input_window, batch_size, d_model)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2) # Shape: (batch_size, input_window, d_model) - back to batch_first

        # 3. Transformer Encoder
        # Output shape: (batch_size, input_window, d_model)
        encoder_output = self.transformer_encoder(src)

        # 4. Multi-horizon Output Heads
        # We need to pool or flatten the encoder_output across the sequence dimension.
        # A simple approach is to flatten the last output of the sequence, but TST often uses the output of the last token
        # or aggregates across the sequence. Given multi-horizon, let's try flattening for now.
        
        # Flatten the output of the encoder for the output layers
        # Shape before flatten: (batch_size, input_window, d_model)
        # Shape after flatten: (batch_size, input_window * d_model)
        flattened_output = encoder_output.view(encoder_output.size(0), -1)

        # Pass flattened output through each horizon's linear layer
        predictions = []
        for output_layer in self.output_layers:
            predictions.append(output_layer(flattened_output))

        # Concatenate predictions from all horizons
        # Shape: (batch_size, num_horizons)
        # Note: This assumes each output layer predicts a single value. Adjust if predicting quantiles.
        final_predictions = torch.cat(predictions, dim=1)

        return final_predictions

# Example Usage (for testing the structure)
# num_features = 23 # Based on your preprocessing output
# input_window = 168 # From config
# prediction_horizons = [24, 168, 720] # From config
# 
# model = BTCTransformer(
#     num_features=num_features,
#     input_window=input_window,
#     prediction_horizons=prediction_horizons
# )
# 
# # Create a dummy input tensor (batch_size, input_window, num_features)
# dummy_input = torch.randn(32, input_window, num_features)
# 
# # Get predictions
# output = model(dummy_input)
# 
# print("Model output shape:", output.shape) # Expected: (32, len(prediction_horizons)) -> (32, 3) 