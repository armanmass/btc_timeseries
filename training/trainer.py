import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# Assuming data loader and model classes will be imported
# from data.btc_preprocessor import BTCDataLoader
# from models.btc_transformer import BTCTransformer

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
        
        # Initialize GradScaler for AMP
        self.scaler = GradScaler('cuda') if use_amp else None

    def train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        
        # Reset gradients every accumulation steps
        self.optimizer.zero_grad()
        
        for i, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Use autocast for mixed precision if enabled
            with autocast('cuda', enabled=self.use_amp):
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
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            running_loss += loss.item() * self.gradient_accumulation_steps # Unscale loss for reporting

        # Handle remaining gradients if dataset size is not a multiple of accumulation steps
        if len(self.train_loader) % self.gradient_accumulation_steps != 0:
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
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast('cuda', enabled=self.use_amp):
                     outputs = self.model(inputs)
                     loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()

        print(f"Epoch {epoch+1}, Validation Loss: {running_loss / len(self.val_loader)}")
        return running_loss / len(self.val_loader)

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
# if __name__ == "__main__":
#     # Assuming you have model, data_loaders, optimizer, criterion, and device defined
#     # model = BTCTransformer(...).to(device)
#     # train_loader = BTCDataLoader(...).get_train_loader()
#     # val_loader = BTCDataLoader(...).get_val_loader()
#     # optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     # criterion = nn.MSELoss()
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     
#     # trainer = Trainer(
#     #     model=model,
#     #     train_loader=train_loader,
#     #     val_loader=val_loader,
#     #     optimizer=optimizer,
#     #     criterion=criterion,
#     #     device=device,
#     #     epochs=10,
#     #     gradient_accumulation_steps=4, # Example
#     #     use_amp=True
#     # )
#     
#     # trainer.train() 