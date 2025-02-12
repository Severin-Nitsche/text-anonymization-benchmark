import json
from echr_labeling import privacy
from reddit_labeling import r_privacy

def process_echr(path, label=privacy):
    """
    This function converts the ehcr files to usable examples for the longformer.
    It takes a function to label the individual annotations
    """
    with open(path, 'r', encoding='utf-8') as file:
        raw = json.load(file)
        processed = [] # TODO: optimize, this feels terrible

        for data in raw:
            processed.extend([{
                'split': data['dataset_type'],
                'text': data['text'],
                'doc_id': data['doc_id'],
                'annotations': list(filter(
                    lambda annotation: annotation['label'] == 'MASK',
                    [label(annotation, data) for annotation in data['annotations'][annotator]['entity_mentions']]
                ))
            } for annotator in data['annotations']])
        return processed

def process_reddit(path, label=r_privacy):
    """
    This function works similar to process_echr but works with the data generated by the labelling tool for my reddit data
    """
    with open(path, 'r', encoding='utf-8') as file:
        raw = json.load(file)
        processed = []

        for data in raw:
            processed.extend([{
                # 'split': data['dataset_type'],
                'text': data['data']['text'],
                'doc_id': f"reddit-{data['id']}",
                'annotations': list(filter(
                    lambda annotation: annotation['label'] == 'MASK',
                    [label(annotation, data) for annotation in annotator['result']]
                ))
            } for annotator in data['annotations']])
        return processed


dev_processed = process_echr('echr_dev.json')
train_processed = process_echr('echr_train.json')
test_processed = process_echr('echr_test.json')