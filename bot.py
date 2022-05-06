import os
import logging

import discord
from dotenv import load_dotenv

from utils.colorizer import colorize
import bot.events as events
from utils.configReader import readConfig
from utils.setupLogger import setupLogger

load_dotenv()

config = readConfig()

logger = setupLogger('ai', level=config['bot']['logLevel'])
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
