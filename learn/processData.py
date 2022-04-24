import glob
import regex as re

import pandas as pd

from utils.configReader import readConfig

config = readConfig()

class DataProcessor:
	def __init__(self):
		self.dataframe = pd.DataFrame()

	def run(self):
		self.importData()
		self.dropColumns()
		self.dropAttachments()
		self.filterContent()
		self.dropEmptyContent()
		return self.dataframe

	def importData(self):
		listOfDFs = []
		for fileGlob in config['training']['data']['inputFiles']:
			for fileName in glob.glob(f'data/{fileGlob}'):
				listOfDFs.append(pd.read_csv(fileName))
		self.dataframe = pd.concat(listOfDFs)

	def dropColumns(self):
		if 'ChannelID' in self.dataframe.columns:
			self.dataframe.drop('ChannelID', axis=1, inplace=True)
		if 'AuthorID' in self.dataframe.columns:
			self.dataframe.drop('AuthorID', axis=1, inplace=True)
		if 'ID' in self.dataframe.columns:
			self.dataframe.drop('ID', axis=1, inplace=True)
		if 'Timestamp' in self.dataframe.columns:
			self.dataframe.drop('Timestamp', axis=1, inplace=True)
	
	'''Drop columns and filter some stuff according to config'''
	def dropAttachments(self):
		self.dataframe = self.dataframe[self.dataframe['Attachments'].isnull()]
		self.dataframe.drop('Attachments', axis=1, inplace=True)

	@staticmethod
	def filterString(string):
		if not isinstance(string, str):
			return string
		# filter discord emotes
		if config['training']['data']['filterDiscordEmotes']:
			string = re.sub(r'<:[a-zA-Z0-9_]+?:\d{18}>', '', string)
		# filter emoji
		# FIXME check why this isnt working
		if config['training']['data']['filterVanillaEmoji']:
			string = re.sub(r'(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])', '', string)
		# filter pings
		if config['training']['data']['filterPings']:
			string = re.sub(r'<@!?\d{18}>', '', string)
		# filter channels
		if config['training']['data']['filterChannels']:
			string = re.sub(r'<#\d{18}>', '', string)
		# filter links
		if config['training']['data']['filterLinks']:
			string = re.sub(r'<?https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)>?', '', string)
		# filter markdown
		if config['training']['data']['filterMarkdown']:
			string = re.sub(r'(\*\*\*(?=[^\n]*\*\*\*))|((?<=\*\*\*[^\n]*)\*\*\*)', '', string) # bold italic
			string = re.sub(r'(\*\*(?=[^\n]*\*\*))|((?<=\*\*[^\n]*)\*\*)', '', string) # bold
			string = re.sub(r'(\*(?![ \n])(?=[^\n]*\*))|((?<=\*[^ ][^\n]*)\*)', '', string) # italic
			string = re.sub(r'(~~(?=[^\n]*~~))|((?<=~~[^\n]*)~~)', '', string) # strikethrough
			string = re.sub(r'(\_\_(?=[^\n]*\_\_))|((?<=\_\_[^\n]*)\_\_)', '', string) # underline
			string = re.sub(r'(\|\|(?=[^\n]*\|\|))|((?<=\|\|[^\n]*)\|\|)', '', string) # spoiler
			string = re.sub(r'(```(?=[^\n]*```))|((?<=```[^\n]*)```)', '', string) # code block
			string = re.sub(r'(`(?=[^\n]*`))|((?<=`[^\n]*)`)', '', string) # inline code
		# TODO add custom filtered chars
		return string
	
	def filterContent(self):
		self.dataframe['Contents'] = self.dataframe['Contents'].apply(self.filterString)

	def dropEmptyContent(self):
		self.dataframe.dropna(subset=['Contents'], inplace=True)
		self.dataframe.drop(self.dataframe[self.dataframe['Contents'] == ''].index, inplace=True)
	

	
