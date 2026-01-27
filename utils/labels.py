import yaml

def get_labels_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data.get('labels', [])

def ishuman(label):
    return label in ["human", "girl", "man", "woman", "old woman", "boy", "old man", "person", "people"]