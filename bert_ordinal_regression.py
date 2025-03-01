import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from coral_pytorch.losses import corn_loss
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from torch.utils.data import random_split
import pickle


class hfDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

def split_dataset_torch(dataset, val_size=0.2):
    val_size = int(len(dataset) * val_size)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset
    


def build_dataset(dataset_path):
    with open(dataset_path,"rb") as input_file:
        ds = pickle.load(input_file)
        

    ds = hfDataset(ds["encodings"], ds["labels"])
    

    return ds

class DistilBertOrdinal(nn.Module):
    def __init__(self, num_classes):
        super(DistilBertOrdinal, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes-1)  # K-1 logits

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    


class BERTModel:
    
    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.model = None

    def build_model(self,num_classes):
        self.model = DistilBertOrdinal(num_classes=num_classes)
        self.model.to(self.device)

    def load_model(self,model_path):
        self.model.load_state_dict(torch.load(model_path))
        


    def train_model(self,dataset_path, num_classes=4, batch_size=8, num_epochs=3, lr=5e-5):
        # Create DataLoader
        # Load the dataset from the pickle file
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)  # data is a dictionary with "encodings" and "class"

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_dict({
            "input_ids": data["encodings"]["input_ids"],
            "attention_mask": data["encodings"]["attention_mask"],
            "label": data["class"]
        })

        # Set format for PyTorch
        dataset.set_format(type="torch")
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            print("Batch size:", batch["input_ids"].size(0), batch["label"].size(0))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                if batch["input_ids"].size(0) == 0 or batch["label"].size(0) == 0:  # Skip empty batches
                    print(batch["input_ids"], batch["label"])
                    continue
                else:
                    optimizer.zero_grad()
                    logits = self.model(batch["input_ids"],batch["attention_mask"])
                    try:
                        loss = corn_loss(logits, batch["label"], num_classes)
                    except:
                        print(f"Division by zero. Skipping batch.")
                        print(f"Batch size: {batch['input_ids'].size(0)}")
                        print(f"Labels: {batch['label']}")
                        continue
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
        


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)



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
            