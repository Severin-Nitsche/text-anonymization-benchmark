"""
The following contains a function that can be used with reddit data
"""

def r_privacy(annotation, data):
    """
    This function masks every PII as well as private information
    """
    mask = set(['BELIEF', 'POLITICS', 'SEX', 'ETHNIC', 'HEALTH', 'DIRECT_ID', 'QUASI_ID'])
    return {
        'start_offset': annotation['value']['start'],
        'end_offset': annotation['value']['end'],
        'span_text': annotation['value']['text'],
        'label': 'MASK' if bool(mask & set(annotation['value']['labels'])) else 'NO_MASK',
        'id': data['id']
    }