import torch
from torch import nn
from torch.optim import AdamW
from transformers import LongformerTokenizerFast
from toy_data_handling import *
from torch.utils.data.dataloader import DataLoader
from longformer_model import Model
from data_manipulation import train_processed, test_processed
import tqdm

# With thanks from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_one_epoch(model, optimizer, loss_fn, train_loader):
    running_loss = 0
    last_loss = 0

    for X in tqdm.tqdm(train_loader):
        labels = X['labels']
        optimizer.zero_grad()
        outputs = model(X).permute(0,2,1)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        last_loss = loss.item()
        running_loss += last_loss
    return last_loss, running_loss / len(train_loader)

if __name__ == '__main__':
    bert = 'allenai/longformer-base-4096'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')

    tokenizer = LongformerTokenizerFast.from_pretrained(bert)
    label_set = LabelSet(labels=["MASK"])

    model = Model(model = bert, num_labels = label_set.num_labels())
    model.to(device)

    if device == 'cuda':
        criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor([1.0, 10.0, 10.0]).cuda())
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor([1.0, 10.0, 10.0]))
    
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    train = WindowedDataset(data=train_processed, tokenizer=tokenizer, label_set=label_set, include_annotations=True, tokens_per_batch=4096)
    trainloader = DataLoader(train, collate_fn=WindowBatch, batch_size=1)
    test = WindowedDataset(data=test_processed, tokenizer=tokenizer, label_set=label_set, include_annotations=True, tokens_per_batch=4096)
    testloader = DataLoader(test, collate_fn=WindowBatch, batch_size=1)

    losses, epochs = [], []
    for epoch in range(5):
        epochs.append(epoch)
        model.train()
        last_loss, avg_loss = train_one_epoch(model, optimizer, criterion, trainloader)
        print('Epoch: ', epoch + 1)
        print('Training loss (last/avg): {0:.2f}/{0:.2f}'.format(last_loss, avg_loss))
    
    model.eval()
    running_loss = 0
    for X in tqdm.tqdm(testloader):
        labels = X['labels']
        outputs = model(X).permute(0,2,1)

        loss = criterion(outputs, labels)

        running_loss += loss.item()
    avg_loss = running_loss / len(trainloader)
    print('Validation loss: {0:.2f}'.format(avg_loss))

    PATH = "long_model.pt"
    torch.save(model.state_dict(), PATH)
