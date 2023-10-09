import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import csv
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
submissionPath = "my_submission.csv"
trainPath = "smishing_train.txt"
validPath = "smishing_valid.txt"
testPath = "smishing_test.txt"
model_name = "twitter-roberta-base"
checkpoint_path = "best_checkpoint.pt"


# 1. 读取文本数据制作成pytorch专用数据集
class TextDataset(Dataset):
    def __init__(self, data_file):
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for line in reader:
                self.data.append((line[0], int(line[1])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        tokenized = tokenizer(text, truncation=True, padding='max_length', 
                              max_length=128, return_tensors="pt")
        return tokenized, label

class TextDatasetTest(Dataset):
    def __init__(self, data_file):
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for line in reader:
                self.data.append((line[0], 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        tokenized = tokenizer(text, truncation=True, padding='max_length', 
                              max_length=128, return_tensors="pt")
        return tokenized, label

# 2. 对句子进行分词得到词库
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 3. 构建DataLoader
train_dataset = TextDataset(trainPath)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

validation_dataset = TextDataset(validPath)
validation_loader = DataLoader(validation_dataset, batch_size=128)


# test_dataset = TextDataset(testPath)
# test_loader = DataLoader(test_dataset, batch_size=128)
# load test data which is not labeled
test_dataset = TextDatasetTest(testPath)
test_loader = DataLoader(test_dataset, batch_size=128)

# 4. 定义模型
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
print("before model")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
# model.classifier.out_features = 3


# 5. 训练模型
def train_model(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


    best_val_loss = float('inf')  # Initialize the best validation loss as infinity
    checkpoint_path = "best_checkpoint.pt"

    print("before epoch")
    for epoch in tqdm(range(10)):
        # Training Phase
        model.train()  # Set model to training mode
        total_train_loss = 0
        for i, (text, label) in enumerate(train_loader):
            print("i:", i)
            inputs = {key: val.squeeze().to(device) for key, val in text.items()} # Move input tokens to device
            label = label.to(device)
            optimizer.zero_grad()  # Clear any previously calculated gradients    

            # Forward pass
            outputs = model(**inputs, labels=label)

            loss = outputs.loss
            total_train_loss += loss.item()
        
            # Backward pass
            loss.backward()
            optimizer.step()
    
        average_train_loss = total_train_loss / len(train_loader)
        print("Average train loss:", average_train_loss)
    
        # Validation Phase
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for i, (text, label) in enumerate(validation_loader):
                inputs = {key: val.squeeze().to(device) for key, val in text.items()}
                label = label.to(device)
            
                # Forward pass
                outputs = model(**inputs, labels=label)
                loss = outputs.loss
                total_val_loss += loss.item()
        average_val_loss = total_val_loss / len(validation_loader)
        print("Average validation loss:", average_val_loss)
    
        # Check if this is the best model so far
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            torch.save(model.state_dict(), checkpoint_path)
    


# Step 3: Load the best model for further tasks
model.load_state_dict(torch.load(checkpoint_path))

# hello
# use the best model to make predictions on the test dataset
def make_predictions(model, data_loader):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for i, (text, label) in enumerate(data_loader):
            inputs = {key: val.squeeze().to(device) for key, val in text.items()}
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, dim=1)
            all_predictions.extend(predicted.cpu().numpy())
            print("i:", i)
    return all_predictions

def write_csv_kaggle_sub(fname, Y):
    # fname = file name
    # Y is a list/array with class entries
    
    # header
    tmp = [['Id', 'Prediction']]
    
    # add ID numbers for each Y
    for (i,y) in enumerate(Y):
        tmp2 = [(i+1), y]
        tmp.append(tmp2)
        
    # write CSV file
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(tmp)


write_csv_kaggle_sub("my_submission.csv", make_predictions(model, test_loader)) 