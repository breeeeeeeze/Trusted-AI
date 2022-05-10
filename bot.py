import os

import discord
from dotenv import load_dotenv

from bot.EventHandler import EventHandler
from utils.configReader import readConfig
from utils.setupLogger import setupLogger

load_dotenv()

config = readConfig()

logger = setupLogger('ai', level=config['bot']['logLevel'])
setupLogger('discord')


class TrustedAI(discord.Client):
    def __init__(self):
        super().__init__()
        self.eventHandler = EventHandler(self)

    async def on_ready(self):
        await self.eventHandler.on_ready()

    async def on_message(self, message: discord.Message):
        await self.eventHandler.on_message(message)


client = TrustedAI()

client.run(os.getenv('BOT_TOKEN'))
