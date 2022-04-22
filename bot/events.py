from scraper import Scraper

async def on_message(message):
	Scraper.exportMessage(message)

	if message.content.startswith('ai.ping'):
		await message.channel.send('pong')