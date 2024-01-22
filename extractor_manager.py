import json

settings = "settings.json"
with open(settings, 'r') as file:
    data = json.load(file)
extractor_value = data.get('extractor', 'tf')

if extractor_value == "tf" :
    from feature_extractor_tf import FeatureExtractor
else :
    from feature_extractor_opencv import FeatureExtractor

def getExtractor():
    return FeatureExtractor(), extractor_value