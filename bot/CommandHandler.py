import logging
import datetime
from functools import wraps

import regex as re

from bot.PredictionGetter import PredictionGetter
from utils.colorizer import colorize
from utils.configReader import readConfig

config = readConfig()
logger = logging.getLogger('ai.bot.commandhandler')


class CommandHandler:
    def __init__(self):
        self.lastCommand = datetime.datetime.now()
        self.cooldownTime = config['bot']['commandCooldown']

    # Decorators
    class Decorators:

        @staticmethod
        def ownerCommand(command):
            @wraps(command)
            async def wrapper(*args, **kwargs):
                if (
                    str(args[1].author.id) == config['bot']['ownerID']
                    and str(args[1].channel.id) == config['bot']['ownerChannelID']
                ):
                    return await command(*args, **kwargs)
                return

            return wrapper

        @staticmethod
        def cooldown(command):
            @wraps(command)
            async def wrapper(*args, **kwargs):
                self = args[0]
                if datetime.datetime.now() - self.lastCommand < datetime.timedelta(seconds=self.cooldownTime):
                    return
                self.lastCommand = datetime.datetime.now()
                return await command(*args, **kwargs)

            return wrapper

    # Helpers

    async def processCommand(self, command, message, client, **kwargs):
        if command in ['Decorators', 'processCommand']:
            return
        return await getattr(self, command)(message, client, **kwargs)

    # Commands
    @Decorators.ownerCommand
    async def ping(self, message, _):
        return await message.reply('pong')

    @Decorators.ownerCommand
    async def shutdown(self, message, client):
        await message.channel.send('Shutting down...')
        return await client.close()

    @Decorators.cooldown
    async def predict(self, message, _, modelName=None):
        if str(message.channel.id) not in config['bot']['predictor']['predictChannelIDs']:
            return
        splitMessage = message.content.split(' ')[1:]
        if not modelName:
            modelName = splitMessage[0]
            try:
                splitMessage = splitMessage[1:]
            except KeyError:
                splitMessage = None
        temperature = 1.0
        if splitMessage:
            if splitMessage[0] == '-t' or splitMessage[0] == '--temperature':
                temperature = float(splitMessage[1])
                splitMessage = splitMessage[2:]
        if not splitMessage:
            splitMessage.append('\n')
        seed = ' '.join(splitMessage)
        try:
            prediction = await PredictionGetter.predict(modelName, seed, temperature)

            if re.match(r'[a-zA-Z0-9_]+:\d{18}>', prediction):
                prediction = '<:' + prediction
            if re.match(r'@', prediction):
                prediction = '<' + prediction
            for word in config['prediction']['bannedWords']:
                if word in prediction:
                    return await message.reply(config['bot']['strings']['commandHandler.bannedWord'])
            logger.info(
                (
                    f'{colorize(f"{message.author.name}#{message.author.discriminator}","OKBLUE")}'
                    f' requested prediction from {colorize(modelName, "OKGREEN")}'
                    f' model: {colorize(prediction, "OKCYAN")}'
                )
            )
            return await message.reply(prediction)
        except Exception as err:
            logger.error(colorize(err, 'FAIL'))
            return await message.reply(config['bot']['strings']['commandHandler.predictionError'])
