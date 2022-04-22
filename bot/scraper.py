import datetime
import json
import os
import logging
from colorer import colorize

def loadConfig():
	with open('../config.json') as f:
		return json.load(f)
config = loadConfig()

logger = logging.getLogger('discord.Scraper')

class Scraper:

	@classmethod
	def exportMessage(cls, message):
		try:
			if not str(message.guild.id) == config['serverID'] \
				or not str(message.channel.id) in config['channelIDs'] \
				or message.author.bot \
				or not message.content \
				or message.content.startswith('ai.'): 
				return
			fileName = Scraper.getFileName()
			if not os.path.exists(fileName):
				with open(fileName, 'w') as f:
					f.write('message_content,channel_id,author_id,has_attachment\n')
			with open(fileName, 'a', encoding='utf-8') as f:
				f.write(f'"{message.content}",{str(message.channel.id)},{str(message.author.id)},{str(bool(message.attachments))}\n')
			return logger.log(logging.INFO, f'Message logged by {colorize(f"{message.author.name}#{message.author.discriminator}", "OKBLUE")} in {colorize(f"#{message.channel.name}", "OKCYAN")}')
		except Exception:
			return logger.log(logging.WARNING, colorize(f'Failed to export message: {message.id}', 'WARNING'))

	@staticmethod
	def getFileName():
		return config['exportFile'].replace('{date}', datetime.datetime.now().strftime('%Y-%m-%d'))