import logging

import discord

from bot.MessageLogger import MessageLogger
from bot.CommandHandler import CommandHandler
from utils.configReader import readConfig
from utils.colorizer import colorize

config = readConfig()
logger = logging.getLogger('ai.bot.events')


class EventHandler:
    def __init__(self, client: discord.Client) -> None:
        self.client = client
        self.commandHandler = CommandHandler()
        self.commandPrefix: str = config['bot']['commandPrefix']
        if config['bot']['messageLogger']['activateMessageLogger']:
            MessageLogger.activate()

    async def on_ready(self) -> None:
        if self.client.user is not None:
            logger.log(logging.INFO, colorize(f'Logged in as {self.client.user.name}', 'OKGREEN'))
        await self.client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name='you'))

    async def on_message(self, message: discord.Message) -> None:
        channelID = str(message.channel.id)
        if not str(message.guild.id) == config['bot']['serverID'] or message.author.bot:  # type: ignore
            return

        if channelID in config['bot']['messageLogger']['messageLoggerChannelIDs']:
            MessageLogger.exportMessage(message)

        # command handler
        if message.content.startswith(self.commandPrefix):
            logger.debug(f'Event: on_message: {message.content}')
            command = message.content.split(' ')[0][len(self.commandPrefix) :]
            # first try to find this command
            try:
                return await self.commandHandler.processCommand(command, message, self.client)
            except Exception:
                pass
            # then try to predict with command as model name
            try:
                assert command in [model['name'] for model in config['prediction']['models']]
                return await self.commandHandler.processCommand('predict', message, self.client, modelName=command)
            except Exception:
                logger.warning(colorize(f'Unknown commmand: {command}', 'WARNING'))
