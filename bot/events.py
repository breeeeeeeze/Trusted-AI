import datetime
import logging

import discord

from bot.scraper import Scraper
from bot.CommandHandler import Commands
from bot.PredictionGetter import PredictionGetter
from utils.configReader import readConfig
from utils.colorizer import colorize

config = readConfig()
logger = logging.getLogger('ai.bot.events')

lastCommand = datetime.datetime.now()


def cooldown():
    global lastCommand  # TODO: make this non-stupid
    if datetime.datetime.now() - lastCommand < datetime.timedelta(
        seconds=config['bot']['commandCooldown']
    ):
        return True
    lastCommand = datetime.datetime.now()
    return False


async def on_ready(client):
    await client.change_presence(
        activity=discord.Activity(type=discord.ActivityType.watching, name='you')
    )
    await PredictionGetter.makeModels()
    logger.info(colorize('AI READY', 'OKGREEN'))


async def on_message(client, message):
    channelID = str(message.channel.id)
    if not str(message.guild.id) == config['serverID'] or message.author.bot:
        return

    if channelID in config['scraperChannelIDs']:
        Scraper.exportMessage(message)

    if channelID not in config['predictChannelIDs']:
        return

    if message.content == 'ai.shutdown' and str(message.author.id) == config['ownerID']:
        await message.channel.send('Shutting down...')
        return await client.close()

    # command handler
    if message.content.startswith('ai.'):
        logger.debug(f'Event: on_message: {message.content}')
        command = message.content.split(' ')[0][3:]
        # first try to find this command
        try:
            func = getattr(Commands, command)
            if cooldown():
                return
            await func(message)
        except Exception:
            pass
        # then try to predict with command as model name
        try:
            assert command in [
                model['name'] for model in config['prediction']['models']
            ]
            func = getattr(Commands, 'predict')
            if cooldown():
                return
            await func(message, command)
        except Exception:
            logger.warning(colorize(f'Unknown commmand: {command}', 'WARNING'))
