from bot.scraper import Scraper
from utils.configReader import readConfig

config = readConfig()


async def on_message(message):
    if not str(message.guild.id) == config['serverID'] \
            or not str(message.channel.id) in config['channelIDs'] \
            or message.author.bot:
        return
    Scraper.exportMessage(message)

    if message.content.startswith('ai.ping'):
        await message.channel.send('pong')
