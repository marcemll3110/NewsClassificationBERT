from torch.utils.data import DataLoader
from transformers import  DistilBertModel, DistilBertPreTrainedModel
import torch
import pickle
import os
from transformers import DistilBertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

import os

class RegressionDataset(Dataset):
    """
    A custom Dataset class for regression tasks using BERT.
    Args:
        input_ids (list or numpy array): List or array of input token IDs.
        attention_mask (list or numpy array): List or array of attention masks.
        labels (list or numpy array): List or array of target labels.
    Attributes:
        input_ids (torch.Tensor): Tensor of input token IDs.
        attention_mask (torch.Tensor): Tensor of attention masks.
        labels (torch.Tensor): Tensor of target labels.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a dictionary containing input_ids, attention_mask, and labels for the given index.
    """

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
    """
    DistilBertForRegression is a custom model for regression tasks using the DistilBERT architecture.
    Args:
        config (PretrainedConfig): Configuration object for DistilBERT.
    Attributes:
        distilbert (DistilBertModel): The DistilBERT model.
        regression_head (nn.Sequential): A sequential container of layers for regression.
    Methods:
        forward(input_ids, attention_mask=None, labels=None):
            Performs a forward pass of the model.
            Args:
                input_ids (torch.Tensor): Tensor of input token IDs.
                attention_mask (torch.Tensor, optional): Tensor of attention masks.
                labels (torch.Tensor, optional): Tensor of labels for training.
            Returns:
                torch.Tensor: The regression logits, clamped between 0 and 1.
    """

    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_size,64),
            nn.LayerNorm(64),
            nn.Dropout(0.4),
            nn.GELU(), 
            nn.Linear(64,1)

        )
        
        self.init_weights()  # Initialize weights properly

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
    
        logits = self.regression_head(pooled_output)  # Pass through regression head
        logits = torch.clamp(logits,0,1) # We clamp the output between 0 and 1
        return logits




class BERTModel:
    """
    A class used to represent a BERT-based regression model.
    Attributes:
  
        device (str): The device to run the model on ('cuda' or 'cpu').
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        model (transformers.DistilBertForRegression): The DistilBERT model for regression tasks.
    Methods:
        __init__(name, device):
            Initializes the BERTModel with the given name and device.
        predict_message(pred_text):
            Predicts the output for a given input text using the pre-trained DistilBERT model.
        train_regression_model(dataset_path, train_ratio=0.8, batch_size=32, learning_rate=1e-4, num_epochs=10, checkpoint_dir='checkpoints', warmup_ratio=0.1, device='cuda' if torch.cuda.is_available() else 'cpu'):
            Trains the regression model using the provided dataset.
        load_checkpoint(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
            Loads a checkpoint to resume training.
    """
    
    def __init__(self, name ,device):
        self.name = name
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.model = DistilBertForRegression.from_pretrained("distilbert-base-uncased")
        self.model.to(self.device)

        

    def predict_message(self,pred_text):
        """
        Predict the output for a given input text using a pre-trained DistilBERT model.
        Args:
            pred_text (str or list): The input text for which the prediction is to be made.
        Returns:
            torch.Tensor: The model's output predictions for the input text.
        """

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        inputs = tokenizer(pred_text, return_tensors='pt', truncation=True, padding=True)
        

        # Move inputs to the same device as the model
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

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
        Train a regression model using the provided data dictionary. Stores the last epoch and the one that had the smallest validation loss.
        
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
        self.optimizer = optimizer
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
        self.scheduler = scheduler
        
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
                'scheduler_state_dict': scheduler.state_dict(),  
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            
            # Save checkpoint if it's the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f'New best validation loss: {best_val_loss:.4f}. Saving checkpoint...')
                torch.save(checkpoint, f'{checkpoint_dir}/{self.name}_smallest_val_loss.pt')
            
            # Save last epoch checkpoint
            if epoch == num_epochs - 1:
                torch.save(checkpoint, f'{checkpoint_dir}/{self.name}_last_epoch.pt')
        
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
        self.optimizer = optimizer
        
        # Create a dummy scheduler (proper parameters needed if continuing training)
        scheduler = OneCycleLR(optimizer, max_lr=1e-4, total_steps=1000)  # Placeholder values
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scheduler = scheduler
        print(f"Loaded trained model: {checkpoint_path}")
        

                


            

