from pathlib import Path
import json
import spacy
from spacy.language import Language

def load_prompts_from_msmarco_samples_from_rag_truth(dataset_path: Path) -> list[str]:
    source_json_path = dataset_path / "source_info.jsonl"

    source_info_data = []
    with open(source_json_path, 'r') as json_file:
        for json_str in json_file:
            cur_data = json.loads(json_str)
            source_info_data.append(cur_data)

    marco_samples = [sample for sample in source_info_data if sample['source'] == 'MARCO']
    prompts = [sample["prompt"] for sample in marco_samples]

    return prompts


@Language.component("custom_splits")
def custom_splits(doc):
    for token in doc[:-1]:
        if token.text == '\n' or token.text == '\n\n' or token.text in (','):
            doc[token.i + 1].is_sent_start = True
    return doc


class SentenceSplitter:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("custom_splits", before="parser")


    def get_sentence_boundaries(self, text: str) -> list[tuple[int, int]]:
        doc = self.nlp(text)
        sentence_boundaries = [(sent.start_char, sent.end_char) for sent in doc.sents]
        
        return sentence_boundaries


def insert_brackets(text, positions):
    # Sort positions by start index in descending order to avoid messing up indices
    positions.sort(key=lambda x: x[0], reverse=True)
    
    for start, end in positions:
        text = text[:end] + ']' + text[end:]
        text = text[:start] + '[' + text[start:]
    
    return text