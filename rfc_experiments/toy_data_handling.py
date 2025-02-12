from typing import List, Any
from tokenizers import Encoding
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import torch

Tokens = List[int]
Batch = List[Tokens]

def align_tokens_and_annotations_bio2(tokenized: Encoding, annotations):
    """
    Given `annotations` [{label: <label>, start_offset: <offset>, end_offset: <offset>}]
    computes BIO2 alignment with tokens in `tokenized` [See](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging))
    returns `aligned_labels` where `aligned_labels[i]` [BIO][-<label-of-token-i>]
    requires ...TokenizerFast to access token indices in raw text
    """
    tokens = tokenized.tokens
    aligned_labels = ["O"] * len(tokens) # Initially everything is outside a label
    for annotation in annotations:
        first_token = float('inf')
        for char_ix in range(annotation["start_offset"], annotation["end_offset"]):
            token_ix = tokenized.char_to_token(char_ix)
            if token_ix is not None:
                first_token = min(first_token, token_ix)
                aligned_labels[token_ix] = f"I-{annotation['label']}"
        aligned_labels[first_token] = f"B-{annotation['label']}"
    return aligned_labels # TODO: data_handling.py returned identifier_types, offsets, ids where ids is an additional argument to this function. Check this.

class LabelSet:
    """
    Utility class that converts between textual BIO(2)-prefixed labels and integer ids.
    Given unprefixed labels.
    """
    def __init__(self, labels: List[str]):
        self.labels = labels
    def num_labels(self):
        """
        Returns the number of labels
        """
        return 2 * len(self.labels) + 1
    def id_to_label(self, id):
        """
        Convert from id to label
        0: O, 1: B-<label-1>, 2: I-<label-1>, 3: B-<label-2>, 4: I-<label-2>, ...
        """
        if id == 0:
            return "O"
        else:
            return f'{"BI"[(id-1)%2]}-{self.labels[(id-1)//2]}'
    def id_to_color(self, id):
        """
        ANSI COLOR CODES FOR LABELS
        """
        if id == 0:
            return "\033[0m"
        else:
            return '\033[93m'
    def label_to_id(self, label):
        """
        Convert from label to id
        (See id_to_label for format)
        """
        if label == 'O':
            return 0
        elif label[0] == 'B':
            return 1 + 2*self.labels.index(label[2:])
        else:
            return 2 + 2*self.labels.index(label[2:])
    def get_aligned_label_ids_from_annotations(self, tokenized_text, annotations):
        """
        Converts the textual alignment of align_tokens_and_annotations_bio2
        to an alignment of ids
        (Takes the same input as the referenced function)
        """
        text_labels = align_tokens_and_annotations_bio2(tokenized_text, annotations)
        return list(map(self.label_to_id, text_labels)) # TODO: In data_handling.py identifier_types, offsets, ids (See align_tokens_and_annotations_bio2)
    def pretty_print_token_tags(self, text, offsets, labels):
        for (start, end), label in zip(offsets, labels):
            print(f'{self.id_to_color(label)}{text[start:end]}', end=' ')
        print('\n')

"""
Windowing Logic
"""

@dataclass
class WindowEntry:
    tokens: Tokens # TODO: Is this the tokenized input?
    attention_masks: Tokens # The tokens that are not padding (prevent attending to non-sensical pad values)
    labels: Tokens # Target label ids ([] if inference only)
    offsets: Tokens # Offsets in original text
    ix: int # The data index (Due to windowing, one entry might span multiple windowEntries)

# TODO: here again we exlclude more data from data_handling.py

class WindowedDataset(Dataset):
    """
    Window and padd real-world data to fit our model
    """
    def __init__(
        self,
        data: Any,
        label_set: LabelSet,
        tokenizer: PreTrainedTokenizerFast,
        include_annotations=True,
        tokens_per_batch=32,
        window_stride=None
    ):
        # TODO: Check if everything here needs to be a member
        self.label_set = label_set
        if window_stride is None:
            window_stride = tokens_per_batch
        self.window_stride = window_stride
        self.tokenizer = tokenizer
        # self.texts = []
        # self.annotations = [] # Only relevant for Training

        # for entry in data:
        #     self.texts.append(entry["text"])
        #     if include_annotations:
        #         self.annotations.append(entry["annotations"])

        texts = list(map(lambda entry: entry['text'], data))
        # annotations = map(lambda entry: entry['annotations'], data)

        # Tokenize the data
        tokenized_batch = self.tokenizer(texts, add_special_tokens=False)
        # print(tokenized_batch.offset_mapping)
        # TODO: Again data_handling.py did some offset stuff here (also some offset flag in tokenized_batch)

        # Create a list with windows
        self.entries: List[WindowEntry] = []
        for ix, (encoding, entry) in enumerate(zip(tokenized_batch.encodings, data)):
            # encoding = tokenized_batch[ix]
            sequence_length = len(encoding.tokens)
            labels = [] # padding tokens
            if include_annotations: # Align annotations
                raw_annotations = entry['annotations'] # self.annotations[ix]
                labels = label_set.get_aligned_label_ids_from_annotations(encoding, raw_annotations)
            
            for start in range(0, max(sequence_length - tokens_per_batch + 1,1), self.window_stride):
                end = min(start + tokens_per_batch, sequence_length)
                padding_to_add = max(0, tokens_per_batch - end + start)
                
                self.entries.append(
                    WindowEntry(
                        tokens=encoding.ids[start:end] + [self.tokenizer.pad_token_id] * padding_to_add,
                        labels=labels[start:end] + [-100] * padding_to_add if include_annotations else None, # padded labels
                        attention_masks=encoding.attention_mask[start:end] + [0] * padding_to_add,
                        offsets=encoding.offsets,
                        ix=ix
                    )
                )
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx) -> WindowEntry:
        return self.entries[idx]

"""
Batching logic
Convert from Dataset aka. glorified lists to tensors
(This will automatically use CUDA if available)
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class WindowBatch:
    def __getitem__(self, item):
        return getattr(self, item)
    def __init__(self, entries: List[WindowEntry]):
        self.input_ids: torch.Tensor
        self.attention_masks: torch.Tensor
        self.labels: torch.Tensor
        self.offsets: List
        self.ixs: List

        self.offsets = []
        self.ixs = []

        input_ids: Batch = []
        masks: Batch = []
        labels: Batch = []
        for entry in entries:
            input_ids.append(entry.tokens)
            masks.append(entry.attention_masks)
            labels.append(entry.labels)
            self.offsets.append(entry.offsets)
            self.ixs.append(entry.ix)
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_masks = torch.LongTensor(masks)
        if labels[0] is not None:
            self.labels = torch.LongTensor(labels)
            self.labels.to(device)
        self.input_ids.to(device)
        self.attention_masks.to(device)
# TODO: again we omitted the alignment stuff