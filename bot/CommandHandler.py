import logging

import regex as re

from bot.PredictionGetter import PredictionGetter
from utils.colorizer import colorize
from utils.configReader import readConfig

config = readConfig()
logger = logging.getLogger('ai.bot.commandhandler')


class Commands:

    @staticmethod
    async def ping(message):
        if not str(message.author.id) == config['ownerID']:
            return
        return await message.channel.send('pong')

    @staticmethod
    async def predict(message, modelName=None):
        splitMessage = message.content.strip().split(' ')[1:]
        if not modelName:
            modelName = splitMessage[0]
            splitMessage = splitMessage[1:]
        temperature = 1.0
        if splitMessage[0] == '-t' or splitMessage[0] == '--temperature':
            temperature = float(splitMessage[1])
            splitMessage = splitMessage[2:]
        seed = ' '.join(splitMessage)
        try:
            prediction = await PredictionGetter.predict(modelName, seed, temperature)
            if re.match(r'[a-zA-Z0-9_]+:\d{18}>', prediction):
                prediction = '<:' + prediction
            if re.match(r'@', prediction):
                prediction = '<' + prediction
            for word in config['prediction']['bannedWords']:
                if word in prediction:
                    return await message.channel.send(
                        config['bot']['strings']['commandHandler.bannedWord'])
            logger.info((
                f'{colorize(f"{message.author.name}#{message.author.discriminator}", "OKBLUE")}'
                f' requested prediction from {colorize(modelName, "OKGREEN")}'
                f' model: {colorize(prediction, "OKCYAN")}'))
            return await message.channel.send(prediction)
        except Exception:
            return await message.channel.send(
                config['bot']['strings']['commandHandler.predictionError'])
