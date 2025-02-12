"""
The following contains two example functions that can be used with processing the ehcr data
"""

def privacy(annotation, data):
    """
    This function masks every PII as well as private information
    """
    if annotation['identifier_type'] != 'NO_MASK' or annotation['confidential_status'] != 'NOT_CONFIDENTIAL':
        annotation['label'] = 'MASK'
    else:
        annotation['label'] = 'NO_MASK'
    annotation['id'] = data['doc_id']
    return annotation

def de_identify(annotation, data):
    """
    This is the default function used in the original training
    """
    if annotation['identifier_type'] != 'NO_MASK':
        annotation['label'] = 'MASK'
    else:
        annotation['label'] = 'NO_MASK'
    annotation['id'] = data['doc_id']
    return annotation