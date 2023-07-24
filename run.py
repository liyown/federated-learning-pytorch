import json

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
from models import *

transform = Compose(
    [RandomHorizontalFlip(), ColorJitter(0.5, 0.5), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainDatasets = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
testDatasets = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, target_transform=None,
                                            download=True)

dataLoader = DataLoader(trainDatasets, batch_size=1280, shuffle=True)
testLoader = DataLoader(testDatasets, batch_size=1280, shuffle=False)

model = FedAvgCNN("cifar10").to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
criterion = torch.nn.CrossEntropyLoss().to("cuda")

model.train()
results = {"loss": [], "acc": []}
for e in range(150):
    for data, labels in dataLoader:
        data, labels = data.to("cuda"), labels.to("cuda")
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        loss = 0
        total = 0
        for data, labels in testLoader:
            data, labels = data.to("cuda"), labels.to("cuda")
            outputs = model(data)
            loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("epoch:{},acc:{},loss:{}".format(e, correct / total, loss / total))
        results["loss"].append(loss / total)
        results["acc"].append((correct / total) * 100)

with open("local_dirichlet_1.json", encoding="utf-8", mode="w") as f:
    json.dump(results, f)
