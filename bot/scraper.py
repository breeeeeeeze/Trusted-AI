import datetime
import os
import logging

from utils.colorizer import colorize
from utils.configReader import readConfig

config = readConfig()

logger = logging.getLogger('ai.bot.scraper')


class Scraper:

    @staticmethod
    def exportMessage(message):
        try:
            if not message.content \
                    or message.content.startswith('ai.'):
                return
            fileName = Scraper.getFileName()
            if not os.path.exists(fileName):
                with open(fileName, 'w') as f:
                    f.write('ID,Contents,ChannelID,AuthorID,Attachments\n')
            content = message.content
            channelID = str(message.channel.id)
            authorID = str(message.author.id)
            attachment = str(message.attachments[0]) if message.attachments else ''
            messageID = str(message.id)
            with open(fileName, 'a', encoding='utf-8') as f:
                f.write(f'{messageID},"{content}",{channelID},{authorID},{attachment}\n')
            return logger.log(
                logging.INFO,
                (f'Message logged by '
                    f'{colorize(f"{message.author.name}#{message.author.discriminator}", "OKBLUE")}'  # noqa: E501
                    f' in {colorize(f"#{message.channel.name}", "OKCYAN")}'))
        except Exception:
            return logger.log(
                logging.WARNING,
                colorize(f'Failed to export message: {message.id}', 'WARNING'))

    @staticmethod
    def getFileName():
        return config['exportFile'].replace('{date}', datetime.datetime.now().strftime('%Y-%m-%d'))
