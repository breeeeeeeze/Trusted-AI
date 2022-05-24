import logging
import datetime
import traceback
from functools import wraps
from typing import List, Union

import regex as re
import discord

from bot.PredictionGetter import PredictionGetter
from bot.MessageLogger import MessageLogger
from utils.colorizer import colorize
from utils.configReader import readConfig

config = readConfig()
logger = logging.getLogger('ai.bot.commandhandler')


class CommandHandler:
    """
    Class to handle bot commands
    """
    def __init__(self) -> None:
        self.lastCommand = datetime.datetime.now()
        self.cooldownTime: Union[int, float] = config['bot']['commandCooldown']
        self.activatePredictor: bool = config['bot']['predictor']['activatePredictor']
        self.predictionGetter = PredictionGetter(self.activatePredictor)
        self.strings: dict = config['bot']['strings']

    # Decorators
    class Decorators:
        """
        Decorators that are used in the command handler
        """

        @staticmethod
        def ownerCommand(command):
            """
            Decorator to mark a command as owner only
            """
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
            """
            Decorator to mark a command as having a cooldown. Cooldowns are global, not per-command
            """
            @wraps(command)
            async def wrapper(*args, **kwargs):
                self: CommandHandler = args[0]
                if datetime.datetime.now() - self.lastCommand < datetime.timedelta(seconds=self.cooldownTime):
                    return
                self.lastCommand = datetime.datetime.now()
                return await command(*args, **kwargs)

            return wrapper

    # Helpers

    async def processCommand(self, command: str, message: discord.Message, client: discord.Client, **kwargs):
        """
        Wrapper to process a command while protecting internal functions
        """
        if command in ['Decorators', 'processCommand', 'initPredictionGetter'] or command.startswith('_'):
            return
        return await getattr(self, command)(message, client, **kwargs)

    # Commands
    @Decorators.ownerCommand
    async def ping(self, message: discord.Message, *_) -> None:
        return await message.reply('pong')

    @Decorators.ownerCommand
    async def shutdown(self, message: discord.Message, client: discord.Client) -> None:
        await message.channel.send('Shutting down...')
        return await client.close()

    @Decorators.ownerCommand
    async def logger(self, message: discord.Message, *_) -> None:
        """
        Command for message logger utilities
        """
        try:
            command: str = message.content.split(' ')[1]
        except KeyError:
            return
        if command == 'on':
            MessageLogger.activate()
            return await message.reply(self.strings['messageLogger.activated'], mention_author=False)
        if command == 'off':
            MessageLogger.deactivate()
            return await message.reply(self.strings['messageLogger.deactivated'], mention_author=False)

    @Decorators.ownerCommand
    async def predictor(self, message: discord.Message, *_) -> None:
        """
        Command for predictor utilities
        """
        try:
            command = message.content.split(' ')[1]
        except KeyError:
            return
        if command == 'on':
            reply: discord.Message = await message.reply(self.strings['predictor.activating'], mention_author=False)
            if not self.activatePredictor:
                self.activatePredictor = True
                self.predictionGetter.activate()

            await reply.delete()
            return await message.reply(self.strings['predictor.activated'], mention_author=False)
        if command == 'off':
            self.activatePredictor = False
            self.predictionGetter.deactivate()
            return await message.reply(self.strings['predictor.deactivated'], mention_author=False)

    @Decorators.cooldown
    async def predict(self, message: discord.Message, *_, modelName: str = '') -> None:
        """
        Predict following text based on the seed given, using the model given
        """
        if not self.activatePredictor:
            return
        if str(message.channel.id) not in config['bot']['predictor']['predictChannelIDs']:
            return
        # TODO strip trailing whitespace
        splitMessage: List[str] = message.content.split(' ')[1:]
        if not modelName:
            modelName = splitMessage[0]
            try:
                splitMessage = splitMessage[1:]
            except KeyError:
                splitMessage = []
        temperature = 1.0
        if splitMessage:
            if splitMessage[0] == '-t' or splitMessage[0] == '--temperature':
                temperature = float(splitMessage[1])
                splitMessage = splitMessage[2:]
        if not splitMessage:
            splitMessage.append('\n')
        seed = ' '.join(splitMessage)
        try:
            prediction: str = self.predictionGetter.predict(modelName, seed, temperature)

            if re.match(r'[a-zA-Z0-9_]+:\d{18}>', prediction):
                prediction = '<:' + prediction
            if re.match(r'@', prediction):
                prediction = '<' + prediction
            for word in config['prediction']['bannedWords']:
                if prediction.find(word) != -1:
                    return await message.reply(self.strings['commandHandler.bannedWord'], mention_author=False)
            logger.info(
                (
                    f'{colorize(f"{message.author.name}#{message.author.discriminator}","OKBLUE")}'  # type: ignore
                    f' requested prediction from {colorize(modelName, "OKGREEN")}'
                    f' model: {colorize(prediction, "OKCYAN")}'
                )
            )
            # FIXME prevent sending empty message
            return await message.reply(prediction, mention_author=False)
        except Exception:
            logger.error(colorize(traceback.format_exc(), 'FAIL'))
            return await message.reply(self.strings['commandHandler.predictionError'], mention_author=False)

    @Decorators.cooldown
    async def models(self, message: discord.Message, *_) -> None:
        """
        Command to list all loaded models
        """
        def f(model):
            return model['name'], model['desc']

        modelNames, modelDesc = map(list, zip(*[f(model) for model in config['prediction']['models']]))
        embed = discord.Embed(title='Available models', color=discord.Color.dark_orange())
        if not self.activatePredictor:
            embed.description = 'Predictor is currently inactive. Mald at breeeze to activate.'
        for name, desc in zip(modelNames, modelDesc):
            embed.add_field(name=name, value=desc)
        return await message.reply(embed=embed, mention_author=False)
