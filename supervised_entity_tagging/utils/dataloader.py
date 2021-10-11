import torch
from transformers import PreTrainedTokenizerFast, AutoTokenizer, BatchEncoding
from typing import Optional, List, Tuple, Dict
import os


def _construct_input_batch(sentences: List[List[str]], labels: List[List[str]], tokenizer:PreTrainedTokenizerFast, label2id:Dict[str, int], max_length:int=256) -> Tuple[BatchEncoding, torch.LongTensor]:
    '''
    Note that if you use Roberta, you need to specify "add_prefix_space=True" (just follow the error log)
    '''
    encoded:BatchEncoding = tokenizer(
            text=sentences,
            max_length=max_length,
            is_split_into_words=True,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_special_tokens_mask=False,
            return_tensors='pt'
        )

    label_tensor = - torch.ones_like(encoded.input_ids)

    for ind, sentence_labels in enumerate(labels):
        for word_offset, word_label in enumerate(sentence_labels):
            token_offset = encoded.word_to_tokens(ind, word_offset)
            if token_offset is None:
                continue
            else:
                label_tensor[ind, token_offset.start:token_offset.end] = label2id[word_label]
    return encoded, label_tensor


def _reconstruct_input_labels(encoded:BatchEncoding, predictions:torch.LongTensor, id2label:Dict[int, str]):
    labels = []
    for ind, sentence_predictions in enumerate(predictions):
        sentence_labels = []
        last_word_offset = -1
        for token_offset, token_label in enumerate(sentence_predictions[1:]):
            word_offset = encoded.token_to_word(ind, token_offset+1)
            if word_offset is None or token_label < 0:
                labels.append(sentence_labels)
                sentence_labels = []
                break
            elif word_offset == last_word_offset:
                # this means that the word label is decided by the leading token
                continue
            else:
                sentence_labels += ['O'] * (last_word_offset + 1 - word_offset)
                sentence_labels.append(id2label[token_label.item()])
                last_word_offset = word_offset
    return labels


class EntityDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self, list_of_word_labels:List[Tuple[List[str], List[str]]], model_name="bert-large-cased", label2id:Optional[Dict[str, int]]=None) -> None:
        super(EntityDataset, self).__init__()
        self.data = list_of_word_labels
        if label2id is None:
            label2id = {'O': 0}
            for _, labels in self.data:
                for label in labels:
                    if label not in label2id:
                        label2id[label] = len(label2id)
        self.label2id = label2id
        self.id2label = {v:k for k,v in label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch:List[Tuple[List[str], List[str]]]) -> Tuple[BatchEncoding, torch.LongTensor]:
        sentences = [instance[0] for instance in batch]
        labels = [instance[1] for instance in batch]
        batch = _construct_input_batch(sentences, labels, self.tokenizer, self.label2id)
        return batch

    def dumps_outputs(self, predictions:List[List[str]]):
        assert len(predictions) == len(self.data)
        outputs = ''
        for instance, pred in zip(self.data, predictions):
            if len(pred) < len(instance[0]):
                pred += ['O'] * (len(instance[0])-len(pred))
            outputs += '\n'.join([f"{token}\t{pred_label}" for token, pred_label in zip(instance[0], pred[:len(instance[0])])])
            outputs += '\n\n'
        return outputs


def _load_file(path:str):
    with open(path, "rt") as fp:
        data = []
        instance = []
        for line in fp:
            if len(line.strip()) > 0:
                instance.append(line.strip().split())
            else:
                data.append(
                    (
                        [word for word, _ in instance],
                        [label for _, label in instance]
                    )
                )
                instance = []
        if len(instance) > 0:
            data.append(
                    (
                        [word for word, _ in instance],
                        [label for _, label in instance]
                    )
                )
    return data


def construct_dataloaders(root:str, model_name="bert-large-cased", batch_size:int=4, num_workers:int=4, seed:int=44739242):
    '''
    root: where you put train.txt, dev.txt and test.txt
    '''
    splits = ["train", "dev", "test"]
    paths = {sp: os.path.join(root, f"{sp}.txt") for sp in splits}
    data = {sp: _load_file(path) for sp, path in paths.items()}
    datasets = {"train": EntityDataset(list_of_word_labels=data["train"])}
    datasets["dev"] = EntityDataset(list_of_word_labels=data["dev"], label2id=datasets["train"].label2id)
    datasets["test"] = EntityDataset(list_of_word_labels=data["test"], label2id=datasets["train"].label2id)
    dataloaders = {sp: torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sp == 'train',
        drop_last=False,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed)
        ) for sp, dataset in datasets.items()}
    return dataloaders
