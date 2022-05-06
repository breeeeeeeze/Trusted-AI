import sys

from tqdm import tqdm

from bot.PredictionGetter import PredictionGetter
from utils.configReader import readConfig
from utils.setupLogger import setupLogger

config = readConfig()
logger = setupLogger('ai', level='DEBUG')

if len(sys.argv) < 3:
    print('Usage: python predict.py [model] [seed]')
    sys.exit(1)

name = str(sys.argv[1])
if name not in [model['name'] for model in config['prediction']['models']]:
    print('Invalid model')
    print(
        f'Available models: {[model["name"] for model in config["prediction"]["models"]]}'
    )
    sys.exit(1)

for model in config['prediction']['models']:
    if model['name'] == name:
        modelDict = model

# seed = str(' '.join(sys.argv[2:]))
seed = '\n'

# set up model

model = PredictionGetter.makeModel(**modelDict)
predictions = []
for _ in tqdm(range(100)):
    predictions.append(model.predict(seed))

for el in predictions:
    print(el)
