import datetime
import os
import logging

from utils.colorizer import colorize
from utils.configReader import readConfig

config = readConfig()

logger = logging.getLogger('ai.bot.messagelogger')


class MessageLogger:
    active = None

    @classmethod
    def activate(cls):
        cls.active = True
        logger.info(f'{colorize("MessageLogger", "OKBLUE")} is {colorize("active", "GREEN")}')

    @classmethod
    def deactivate(cls):
        cls.active = False
        logger.info(f'{colorize("MessageLogger", "OKBLUE")} is {colorize("inactive", "RED")}')

    @classmethod
    def exportMessage(cls, message):
        if not cls.active:
            return
        try:
            if not message.content or message.content.startswith('ai.'):
                return
            filename = cls.getFileName()
            if not os.path.exists(filename):
                with open(filename, 'w') as f:
                    f.write('ID,Contents,ChannelID,AuthorID,Attachments\n')
            content = message.content
            channelID = str(message.channel.id)
            authorID = str(message.author.id)
            attachment = str(message.attachments[0]) if message.attachments else ''
            messageID = str(message.id)
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(f'{messageID},"{content}",{channelID},{authorID},{attachment}\n')
            return logger.log(
                logging.INFO,
                (
                    f'Message logged by '
                    f'{colorize(f"{message.author.name}#{message.author.discriminator}", "OKBLUE")}'  # noqa: E501
                    f' in {colorize(f"#{message.channel.name}", "OKCYAN")}'
                ),
            )
        except Exception:
            return logger.log(
                logging.WARNING,
                colorize(f'Failed to export message: {message.id}', 'WARNING'),
            )

    @staticmethod
    def getFileName():
        return config['bot']['messageLogger']['exportFile'].replace(
            '{date}', datetime.datetime.now().strftime('%Y-%m-%d')
        )
