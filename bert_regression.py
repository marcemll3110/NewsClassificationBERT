from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW, DistilBertModel, DistilBertPreTrainedModel
import torch
import pickle
import os
from transformers import DistilBertTokenizer, Trainer, TrainingArguments
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import os

class RegressionDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }



class DistilBertForRegression(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_size,64),
            nn.LayerNorm(64),
            nn.Dropout(0.4),
            nn.GELU(),  # Linear layer
            nn.Linear(64,1)

        )
        
        self.init_weights()  # Initialize weights properly

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
    
        logits = self.regression_head(pooled_output)  # Pass through regression head
        logits = torch.clamp(logits,0,1)
        return logits




class BERTModel:
    
    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.model = DistilBertForRegression.from_pretrained("distilbert-base-uncased")
        self.model.to(self.device)

        

    def predict_message(self,pred_text):
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        inputs = tokenizer(pred_text, return_tensors='pt', truncation=True, padding=True)
        device = "cpu"

        # Move inputs to the same device as the model
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Forward pass to get predictions
        with torch.no_grad():
            outputs = self.model(input_ids,attention_mask)


        return outputs
    
    
    def train_regression_model(self,
    dataset_path,
    train_ratio=0.8,
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=10,
    checkpoint_dir='checkpoints',
    warmup_ratio=0.1,  # Ratio of total steps for warmup
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Train a regression model using the provided data dictionary.
        
        Args:
            model: PyTorch model
            data_dict: Dictionary containing 'encodings' (with 'input_ids' and 'attention_mask') and 'labels'
            train_ratio: Ratio of data to use for training (rest for validation)
            batch_size: Batch size for training
            learning_rate: Maximum learning rate
            num_epochs: Number of training epochs
            warmup_ratio: Ratio of total steps to use for learning rate warmup
            checkpoint_dir: Directory to save model checkpoints
            device: Device to train on ('cuda' or 'cpu')
        
        Returns:
            dict: Training history containing loss values and learning rates
        """
        # Create dataset
        with open(dataset_path, "rb") as data:
            data_dict = pickle.load(data)

        dataset = RegressionDataset(
            data_dict['encodings']['input_ids'],
            data_dict['encodings']['attention_mask'],
            data_dict['labels']
        )
        
        # Split into train and validation sets
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model and optimizer
        model = self.model.to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        # Calculate total steps for the scheduler
        total_steps = len(train_loader) * num_epochs
        
        # Initialize the learning rate scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=warmup_ratio,  # Portion of steps for warmup
            anneal_strategy='cos',   # Cosine annealing after warmup
            cycle_momentum=False     # Don't cycle momentum
        )
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Keep track of best validation loss
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                
                # Gradient clipping (optional but recommended for transformer models)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                history['learning_rates'].append(scheduler.get_last_lr()[0])
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Save history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
            
            # Save checkpoint if it's the best model so far
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f'New best validation loss: {best_val_loss:.4f}. Saving checkpoint...')
                torch.save(checkpoint, f'{checkpoint_dir}/best_model.pt')
            
            # Save last epoch checkpoint
            if epoch == num_epochs - 1:
                torch.save(checkpoint, f'{checkpoint_dir}/last_epoch.pt')
        
        return history

    def load_checkpoint(self, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Load a checkpoint to resume training.
        
        Args:
            model: PyTorch model
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model on
        
        Returns:
            tuple: (model, optimizer, scheduler, epoch, train_loss, val_loss)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer = Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Create a dummy scheduler (proper parameters needed if continuing training)
        scheduler = OneCycleLR(optimizer, max_lr=1e-4, total_steps=1000)  # Placeholder values
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded trained model: {checkpoint_path}")
        

                


            

