import datetime
import os
import logging

import discord

from utils.colorizer import colorize
from utils.configReader import readConfig
from utils.snowflakeConverter import convertSnowflake

config: dict = readConfig()

logger = logging.getLogger('ai.bot.messagelogger')


class MessageLogger:
    active: bool = False

    @classmethod
    def activate(cls) -> None:
        cls.active = True
        logger.info(f'{colorize("MessageLogger", "OKBLUE")} is {colorize("active", "GREEN")}')

    @classmethod
    def deactivate(cls) -> None:
        cls.active = False
        logger.info(f'{colorize("MessageLogger", "OKBLUE")} is {colorize("inactive", "RED")}')

    @classmethod
    def exportMessage(cls, message: discord.Message) -> None:
        if not cls.active:
            return
        try:
            if not message.content or message.content.startswith('ai.'):
                return
            filename = cls.getFileName()
            if not os.path.exists(filename):
                with open(filename, 'w') as f:
                    f.write('ID,Timestamp,ChannelID,AuthorID,Contents,Attachments\n')
            timestamp = convertSnowflake(message.id).strftime('%Y-%m-%d %H:%M:%S.%f%z')
            content: str = message.content.replace('"', '""')
            channelID = str(message.channel.id)
            authorID = str(message.author.id)  # type: ignore
            attachment = str(message.attachments[0]) if message.attachments else ''
            messageID = str(message.id)
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(f'{messageID},{timestamp},"{content}",{channelID},{authorID},{attachment}\n')
            return logger.log(
                logging.INFO,
                (
                    f'Message logged by '
                    f'{colorize(f"{message.author.name}#{message.author.discriminator}", "OKBLUE")}'  # type: ignore # noqa: E501
                    f' in {colorize(f"#{message.channel.name}", "OKCYAN")}'
                ),
            )
        except Exception:
            return logger.log(
                logging.WARNING,
                colorize(f'Failed to export message: {message.id}', 'WARNING'),
            )

    @staticmethod
    def getFileName() -> str:
        return config['bot']['messageLogger']['exportFile'].replace(
            '{date}', datetime.datetime.now().strftime('%Y-%m-%d')
        )
