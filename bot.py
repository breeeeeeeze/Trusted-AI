import os
import logging

import discord
from dotenv import load_dotenv
import tensorflow as tf

from utils.colorizer import colorize
import bot.events as events
from utils.configReader import readConfig

load_dotenv()

config = readConfig()


def setupLogger(name, level=logging.INFO):
    if name == 'tf':
        logger = tf.get_logger()
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '\u001b[35;1m[%(asctime)s]\u001b[0m[%(name)s][%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    return logger


logger = setupLogger('ai', level=logging.DEBUG)
setupLogger('discord')

client = discord.Client()


@client.event
async def on_ready():
    logger.log(logging.INFO, colorize(f'Logged in as {client.user.name}', 'OKGREEN'))
    await events.on_ready(client)


@client.event
async def on_message(message):
    await events.on_message(client, message)

client.run(os.environ['BOT_TOKEN'])
