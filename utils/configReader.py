import json


def readConfig():
    with open('config.json', encoding='utf-8') as f:
        return json.load(f)
