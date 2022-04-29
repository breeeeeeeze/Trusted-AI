import glob
import logging

import regex as re
import pandas as pd
from tqdm import tqdm

from utils.configReader import readConfig
from utils.colorizer import colorize

config = readConfig()
logger = logging.getLogger('ai.learn.dataprocessor')


class DataProcessor:
    def __init__(self):
        self.dataframe = pd.DataFrame()
        self.text = ''
        self.vocab = None

    def processInputData(self):
        self.importData()
        self.dropColumns()
        self.dropAttachments()
        self.filterContent()
        self.dropEmptyContent()
        self.dfToText()
        self.generateVocab()
        logger.info(f'{colorize("Data imported and processed", "OKGREEN")}')
        return self.text, self.vocab

    def getVocab(self):
        if self.vocab:
            return self.vocab
        self.processInputData()
        return self.vocab

    def exportVocab(self, fileName):
        with open(fileName, 'w', encoding='utf-8') as f:
            for word in self.vocab:
                f.write(word)
        logger.debug('Vocabulary exported.')

    def importVocab(self, fileName):
        with open(fileName, 'r', encoding='utf-8') as f:
            self.vocab = list(f.read())
        logger.debug('Vocabulary imported.')
        return self.vocab

    def importData(self):
        listOfDFs = []
        for fileGlob in config['training']['data']['inputFiles']:
            for fileName in glob.glob(f'data/{fileGlob}'):
                listOfDFs.append(pd.read_csv(fileName))
        self.dataframe = pd.concat(listOfDFs)
        if 'Contents' not in self.dataframe.columns:
            raise Exception('Contents column not found in dataframe.')
        logger.debug('Data imported.')

    def dropColumns(self):
        if 'ChannelID' in self.dataframe.columns:
            self.dataframe.drop('ChannelID', axis=1, inplace=True)
        if 'AuthorID' in self.dataframe.columns:
            self.dataframe.drop('AuthorID', axis=1, inplace=True)
        if 'ID' in self.dataframe.columns:
            self.dataframe.drop('ID', axis=1, inplace=True)
        if 'Timestamp' in self.dataframe.columns:
            self.dataframe.drop('Timestamp', axis=1, inplace=True)
        logger.debug('Columns dropped.')

    def dropAttachments(self):
        if config['training']['data']['removeRowIfAttachment']:
            self.dataframe = self.dataframe[self.dataframe['Attachments'].isnull()]
        self.dataframe.drop('Attachments', axis=1, inplace=True)
        logger.debug('Attachments dropped.')

    @staticmethod
    def filterString(string):
        if not isinstance(string, str):
            return string
        if config['training']['data']['onlyLowercase']:
            string = string.lower()
        # filter discord emotes
        if config['training']['data']['filterDiscordEmotes']:
            string = re.sub(r'<a?:[a-zA-Z0-9_]+?:\d{18}>', '', string)
        # filter emoji
        # FIXME check why this isnt working
        if config['training']['data']['filterVanillaEmoji']:
            string = re.sub(r'(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])', '', string)  # noqa: E501
        # filter pings
        if config['training']['data']['filterPings']:
            string = re.sub(r'<@!?\d{18}>', '', string)
        # filter channels
        if config['training']['data']['filterChannels']:
            string = re.sub(r'<#\d{18}>', '', string)
        # filter links
        if config['training']['data']['filterLinks']:
            string = re.sub(
                r'<?https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)>?', '', string)  # noqa: E501
        # filter markdown
        if config['training']['data']['filterMarkdown']:
            string = re.sub(r'(\*\*\*(?=[^\n]*\*\*\*))|((?<=\*\*\*[^\n]*)\*\*\*)', '', string)  # bold italic # noqa: E501
            string = re.sub(r'(\*\*(?=[^\n]*\*\*))|((?<=\*\*[^\n]*)\*\*)', '', string)  # bold
            string = re.sub(r'(\*(?![ \n])(?=[^\n]*\*))|((?<=\*[^ ][^\n]*)\*)', '', string)  # italic # noqa: E501
            string = re.sub(r'(~~(?=[^\n]*~~))|((?<=~~[^\n]*)~~)', '', string)  # strikethrough
            string = re.sub(r'(\_\_(?=[^\n]*\_\_))|((?<=\_\_[^\n]*)\_\_)', '', string)  # underline
            string = re.sub(r'(\|\|(?=[^\n]*\|\|))|((?<=\|\|[^\n]*)\|\|)', '', string)  # spoiler
            string = re.sub(r'(```(?=[^\n]*```))|((?<=```[^\n]*)```)', '', string)  # code block
            string = re.sub(r'(`(?=[^\n]*`))|((?<=`[^\n]*)`)', '', string)  # inline code
        if config['training']['data']['onlyKeepAllowedChars']:
            string = re.sub(rf'[^{config["training"]["data"]["allowedChars"]}]', '', string)  # noqa: E501
        elif config['training']['data']['filterCustomChars']:
            string = re.sub(rf'[{config["training"]["data"]["customChars"]}]', '', string)
        return string

    def filterContent(self):
        self.dataframe['Contents'] = self.dataframe['Contents'].apply(self.filterString)
        logger.debug('Content filtered.')

    def dropEmptyContent(self):
        self.dataframe.dropna(subset=['Contents'], inplace=True)
        self.dataframe.drop(self.dataframe[self.dataframe['Contents'] == ''].index, inplace=True)
        logger.debug('Empty content dropped.')

    def dfToText(self):
        if self.text:
            return
        for row in tqdm(self.dataframe.itertuples(), total=len(self.dataframe)):
            self.text += row.Contents + '\n'
        logger.debug('Dataframe converted to text.')

    def generateVocab(self):
        self.vocab = sorted(set(self.text))
        logger.debug('Vocabulary generated.')
