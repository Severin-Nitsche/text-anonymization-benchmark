import torch
from transformers import LongformerTokenizerFast
from toy_data_handling import *
from torch.utils.data.dataloader import DataLoader
from longformer_model import Model


bert = "allenai/longformer-base-4096"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LongformerTokenizerFast.from_pretrained(bert)
label_set = LabelSet(labels=["MASK"])

model = Model(model = bert, num_labels = label_set.num_labels())
model.to(device)
model.load_state_dict(torch.load("long_model.pt", weights_only = True, map_location = torch.device(device)))
model.eval()

toy_example = [
    {
        'text' : "Maria Cary, the accountant, at Herbe & Co. has cancer."
    },
    {
        'text' : "Jonathan Smith, the bartender, at Jonny's Nightclub has 100 MIL EUR."
    }
]

toy = WindowedDataset(data=toy_example, tokenizer=tokenizer, label_set=label_set, include_annotations=False, tokens_per_batch=4096)
toyloader = DataLoader(toy, collate_fn=WindowBatch, batch_size=1)

for X in toyloader:
    with torch.no_grad():
        pred = model(X)
        pred = pred.permute(0,2,1)
        print(pred)