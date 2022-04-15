import json
import os
import logging
import discord
from dotenv import load_dotenv
from colorer import colorize

load_dotenv()

def loadConfig():
	with open('config.json') as f:
		return json.load(f)
config = loadConfig()

logging.basicConfig(filename=config['logFile'],
						filemode='a',
						level=logging.INFO,
						format='\u001b[35;1m[%(asctime)s]\u001b[0m[%(name)s][%(levelname)s] %(message)s')

client = discord.Client()

def exportMessage(message):
	with open(config['exportFile'], 'a', encoding='utf-8') as f:
		f.write(f'"{message.content}",{str(message.channel.id)},{str(message.author.id)}\n')
	return f'Message logged by {colorize(message.author.name, "OKBLUE")} in {colorize(message.channel.name, "OKCYAN")}'

@client.event
async def on_ready():
	logging.log(logging.INFO, colorize(f'Logged in as {client.user.name}', 'OKGREEN'))
	await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name='you'))

@client.event
async def on_message(message):
	if not str(message.guild.id) == config['serverID'] \
		or not str(message.channel.id) in config['channelIDs'] \
		or message.author.bot \
		or not message.content: 
		return
	try:
		exported = exportMessage(message)
		logging.log(logging.INFO, exported, 'OKGREEN')
	except Exception:
		logging.log(logging.WARNING, colorize(f'Failed to export message: {message.id}', 'WARNING'))

client.run(os.environ['BOT_TOKEN'])