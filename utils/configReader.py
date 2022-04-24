import json

def readConfig():
	with open('config.json') as f:
		return json.load(f)