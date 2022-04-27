import os
import logging

import discord
from dotenv import load_dotenv

from utils.colorizer import colorize
import bot.events as events
from utils.configReader import readConfig

load_dotenv()

config = readConfig()

logger = logging.getLogger('ai')
logger.setLevel(logging.INFO)
handler = logging.FileHandler(filename=config['logFile'])
handler.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.ERROR)
handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s'))
consoleHandler.setFormatter(
    logging.Formatter('\u001b[35;1m[%(asctime)s]\u001b[0m[%(name)s][%(levelname)s] %(message)s'))
logger.addHandler(handler)
logger.addHandler(consoleHandler)

client = discord.Client()


@client.event
async def on_ready():
    logger.log(logging.INFO, colorize(f'Logged in as {client.user.name}', 'OKGREEN'))
    await client.change_presence(
        activity=discord.Activity(type=discord.ActivityType.watching, name='you'))


@client.event
async def on_message(message):
    await events.on_message(message)

client.run(os.environ['BOT_TOKEN'])
