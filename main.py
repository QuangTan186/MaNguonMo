from typing import Union,List
from fastapi import FastAPI
from pydantic import BaseModel
import random
import json
from NeraulNetwork.unit import *
import torch
import uvicorn

from NeraulNetwork.model import NeuralNet
from nltk_unit import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

app = FastAPI()

@app.post("/")
def answer_question(question):
    sentence = tokenize(question)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    output = model.forward(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        return (f"{bot_name}: I do not understand...")

@app.get("/train")
def train():
    import  numpy as np
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from nltk_unit import stem
    

    with open('intents.json', 'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    num_epochs = 10000
    batch_size = 8
    learning_rate = 0.001
    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)


    class ChatDataset(Dataset):

        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

    model = NeuralNet(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cross_loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words
            labels = labels.to(dtype=torch.long)
            outputs = model(words)
            loss = cross_loss(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


    print(f'final loss: {loss.item():.4f}')

    data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    return(f'training complete. file saved to {FILE}')

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
