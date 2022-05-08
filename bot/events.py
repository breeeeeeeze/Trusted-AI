import logging

import discord

from bot.scraper import Scraper
from bot.CommandHandler import CommandHandler
from bot.PredictionGetter import PredictionGetter
from utils.configReader import readConfig
from utils.colorizer import colorize

config = readConfig()
logger = logging.getLogger('ai.bot.events')
commandHandler = CommandHandler()


async def on_ready(client):
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name='you'))
    await PredictionGetter.makeModels()
    logger.info(colorize('AI READY', 'OKGREEN'))


async def on_message(client, message):
    channelID = str(message.channel.id)
    if not str(message.guild.id) == config['bot']['serverID'] or message.author.bot:
        return

    if (
        config['bot']['messageLogger']['activateMessageLogger']
        and channelID in config['bot']['messageLogger']['messageLoggerChannelIDs']
    ):
        Scraper.exportMessage(message)

    # command handler
    if message.content.startswith('ai.'):
        logger.debug(f'Event: on_message: {message.content}')
        command = message.content.split(' ')[0][3:]
        # first try to find this command
        try:
            return await commandHandler.processCommand(command, message, client)
        except Exception:
            pass
        # then try to predict with command as model name
        try:
            assert command in [model['name'] for model in config['prediction']['models']]
            return await commandHandler.processCommand('predict', message, client, modelName=command)
        except Exception:
            logger.warning(colorize(f'Unknown commmand: {command}', 'WARNING'))
