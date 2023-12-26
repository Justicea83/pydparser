from spacy.tokens import Doc
import spacy
from spacy.training import Example
from datasets import load_dataset, get_dataset_split_names
import os
from spacy.tokens import DocBin
import os.path as pth
import logging

logging.basicConfig(level=logging.INFO)


def get_example_chunk(data):
    nlp = spacy.load('en_core_web_sm')
    examples: list[Example] = []
    for line in data:
        predicted = Doc(nlp.vocab, words=line['tags_skill'])
        reference = Doc(nlp.vocab, words=line['tokens'], tags=line['pos'])
        example = Example(predicted, reference)
        examples.append(example)
    return examples


def save_as_spacy_corpus(dest: str = '', dataset='jjzha/green'):
    os.makedirs(dest, exist_ok=True)

    split_names = get_dataset_split_names(dataset)
    for split_name in split_names:
        doc_bin = DocBin()
        data = load_dataset(dataset, split=split_name)
        chunk = get_example_chunk(data)
        for example in chunk:
            doc_bin.add(example.reference)
        doc_bin.to_disk(pth.join(dest, f"{split_name}.spacy"))


if __name__ == "__main__":
    logging.info('Loading training data from huggingface data...')
    save_as_spacy_corpus(dest=pth.join(pth.dirname(__file__), './data/jd'))
