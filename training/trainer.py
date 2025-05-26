import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader # Import DataLoader
from tqdm import tqdm # Import tqdm

# Assuming data loader and model classes will be imported
# from data.btc_preprocessor import BTCDataLoader
from models.btc_transformer import BTCTransformer
from data.btc_preprocessor import load_and_preprocess_data, PreprocessingConfig # Import necessary functions and classes

class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        train_loader, # Placeholder for DataLoader
        val_loader,   # Placeholder for DataLoader
        optimizer: optim.Optimizer,
        criterion: nn.Module, # Loss function
        device: torch.device,
        epochs: int,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = True # Use Automatic Mixed Precision
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        
        # Initialize GradScaler for AMP using the updated syntax
        self.scaler = GradScaler(enabled=use_amp) # Device is handled by autocast context

    def train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        
        # Reset gradients every accumulation steps
        self.optimizer.zero_grad()
        
        # Wrap train_loader with tqdm for a progress bar
        train_loop = tqdm(self.train_loader, leave=False, desc=f"Epoch {epoch+1} Training")
        for i, (inputs, targets) in enumerate(train_loop): # Use train_loop
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Use autocast for mixed precision if enabled with updated syntax
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps # Scale loss for accumulation

            # Backward pass and optimization step with scaler for AMP
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Perform optimization step and scaler update every accumulation steps
            if (i + 1) % self.gradient_accumulation_steps == 0:
                # Apply gradient clipping
                if self.use_amp:
                    # For AMP, unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # You can adjust max_norm
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            running_loss += loss.item() * self.gradient_accumulation_steps # Unscale loss for reporting

            # Update progress bar description with current loss
            train_loop.set_postfix(loss=running_loss / (i + 1))

        # Handle remaining gradients if dataset size is not a multiple of accumulation steps
        if len(self.train_loader) % self.gradient_accumulation_steps != 0:
            # Apply gradient clipping for remaining gradients
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        print(f"Epoch {epoch+1}, Train Loss: {running_loss / len(self.train_loader)}")

    def validate_one_epoch(self, epoch: int):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            # Wrap val_loader with tqdm for a progress bar
            val_loop = tqdm(self.val_loader, leave=False, desc=f"Epoch {epoch+1} Validation")
            for inputs, targets in val_loop: # Use val_loop
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                     outputs = self.model(inputs)
                     loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()

                # Update progress bar description with current loss
                val_loop.set_postfix(loss=running_loss / (val_loop.n + 1)) # Use val_loop.n + 1 for current average

        val_loss = running_loss / len(self.val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
        return val_loss

    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)
            
            # Basic checkpointing (can be expanded)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Validation loss improved. Saving model.")
                # torch.save(self.model.state_dict(), 'best_model.pth') # Uncomment to save
                
            # Basic early stopping (can be expanded)
            # if early_stopping_condition_met: break 
            
        print("Training finished.")

# Example usage (requires data loader and model instances)
if __name__ == "__main__":
    # Initialize device first
    device_instance = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_instance}")

    # Parameters for model initialization (using examples from btc_transformer.py and plan)
    num_features = 23 
    input_window = 168
    prediction_horizons = [24, 168, 720]
    learning_rate = 1e-4 # Using example LR
    batch_size = 32 # Using example batch size for loaders (though loaders are placeholders)

    # Instantiate the model
    model_instance = BTCTransformer(
        num_features=num_features,
        input_window=input_window,
        prediction_horizons=prediction_horizons,
        # Using default values for d_model, nhead, num_encoder_layers, etc.
    ).to(device_instance) # Move model to the selected device

    # Instantiate the optimizer
    optimizer_instance = optim.Adam(model_instance.parameters(), lr=learning_rate) # Using Adam as in original example

    # Instantiate the criterion (loss function)
    criterion_instance = nn.MSELoss() # Using MSELoss as in original example
    criterion_instance.to(device_instance) # Move criterion to the selected device

    # --- Data Loading and DataLoader Initialization ---
    # Define the path to your data file
    data_file_path = "data/btcusd_1-min_data.csv" # Adjust this path as needed

    # Define preprocessing configuration (using example values)
    preprocessing_config = PreprocessingConfig(
        input_window=input_window,
        prediction_horizons=prediction_horizons,
        train_split=0.7,
        val_split=0.15
        # test_split is automatically calculated
    )

    # Load and preprocess data, getting PyTorch Dataset instances
    # The first return value is the full processed dataframe, which we don't need for the DataLoader
    _, train_dataset, val_dataset, _ = load_and_preprocess_data(
        file_path=data_file_path,
        config=preprocessing_config
    )

    # Create PyTorch DataLoaders
    # Use the batch_size defined earlier
    train_loader_instance = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle training data
        drop_last=True # Drop the last incomplete batch
    )
    val_loader_instance = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        drop_last=False # Keep all validation samples
    )
    print("DataLoaders initialized.")
    # ---------------------------------------------------


    trainer = Trainer(
        model=model_instance,
        train_loader=train_loader_instance,
        val_loader=val_loader_instance,
        optimizer=optimizer_instance,
        criterion=criterion_instance,
        device=device_instance,
        epochs=10, # Adjust epochs for a quick test if needed
        gradient_accumulation_steps=4, # Example, adjust as needed
        use_amp=True # Based on your plan, adjust if needed
    )

    trainer.train() 