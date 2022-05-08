import glob
import logging

import regex as re
import pandas as pd

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
        self.sortByTimestamp()
        self.dropColumns()
        self.dropAttachments()
        self.filterContent()
        self.dropEmptyContent()
        self.resetIndex()
        self.dfToText()
        self.generateVocab()
        logger.info(f'{colorize("Data imported and processed", "OKGREEN")}')
        logger.info(f'{colorize("Total messages: " + str(len(self.dataframe)), "OKGREEN")}')
        return self.text, self.vocab

    def slimTo(self, n):
        # useful for debugging
        self.dataframe = self.dataframe[:n]

    def resetIndex(self):
        self.dataframe.reset_index(drop=True, inplace=True)
        logger.debug('Reindexed.')

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
            self.vocab = sorted(set(f.read()))
        logger.debug('Vocabulary imported.')
        return self.vocab

    def sortByTimestamp(self):
        if 'Timestamp' not in self.dataframe.columns:
            logger.error(
                colorize(
                    'Failed to sort dataframe. Timestamp not found in dataframe.',
                    'FAIL',
                )
            )
        self.dataframe.sort_values('Timestamp', inplace=True)
        self.dataframe.drop('Timestamp', axis=1, inplace=True)
        logger.debug('Sorted by timestamp')

    def importData(self):
        listOfDFs = []
        for fileGlob in config['learn']['data']['inputFiles']:
            for fileName in glob.glob(f'data\\{fileGlob}'):
                logger.debug(f'Importing {fileName}')
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
        if config['learn']['data']['removeRowIfAttachment']:
            self.dataframe = self.dataframe[self.dataframe['Attachments'].isnull()]
        if 'Attachments' in self.dataframe.columns:
            self.dataframe.drop('Attachments', axis=1, inplace=True)
        logger.debug('Attachments dropped.')

    @staticmethod
    def filterString(string):
        if not isinstance(string, str):
            return string
        if config['learn']['data']['onlyLowercase']:
            string = string.lower()
        # filter discord emotes
        if config['learn']['data']['filterDiscordEmotes']:
            string = re.sub(r'<a?:[a-zA-Z0-9_]+?:\d{18}>', '', string)
        # filter emoji
        # FIXME check why this isnt working
        if config['learn']['data']['filterVanillaEmoji']:
            string = re.sub(
                r'(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])',  # noqa: E501
                '',
                string,
            )
        # filter pings
        if config['learn']['data']['filterMentions']:
            string = re.sub(r'<@!?\d{18}>', '', string)
        # filter channels
        if config['learn']['data']['filterChannels']:
            string = re.sub(r'<#\d{18}>', '', string)
        # filter links
        if config['learn']['data']['filterLinks']:
            string = re.sub(
                r'<?https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)>?',  # noqa: E501
                '',
                string,
            )
        # filter markdown
        if config['learn']['data']['filterMarkdown']:
            string = re.sub(r'(\*\*\*(?=[^\n]*\*\*\*))|((?<=\*\*\*[^\n]*)\*\*\*)', '', string)  # bold italic
            string = re.sub(r'(\*\*(?=[^\n]*\*\*))|((?<=\*\*[^\n]*)\*\*)', '', string)  # bold
            string = re.sub(r'(\*(?![ \n])(?=[^\n]*\*))|((?<=\*[^ ][^\n]*)\*)', '', string)  # italic
            string = re.sub(r'(~~(?=[^\n]*~~))|((?<=~~[^\n]*)~~)', '', string)  # strikethrough
            string = re.sub(r'(\_\_(?=[^\n]*\_\_))|((?<=\_\_[^\n]*)\_\_)', '', string)  # underline
            string = re.sub(r'(\|\|(?=[^\n]*\|\|))|((?<=\|\|[^\n]*)\|\|)', '', string)  # spoiler
            string = re.sub(r'(```(?=[^\n]*```))|((?<=```[^\n]*)```)', '', string)  # code block
            string = re.sub(r'(`(?=[^\n]*`))|((?<=`[^\n]*)`)', '', string)  # inline code
        if config['learn']['data']['onlyKeepAllowedChars']:
            string = re.sub(rf'[^{config["learn"]["data"]["allowedChars"]}]', '', string)
        elif config['learn']['data']['filterCustomChars']:
            string = re.sub(rf'[{config["learn"]["data"]["bannedChars"]}]', '', string)
        return string

    def filterContent(self):
        self.dataframe['Contents'] = self.dataframe['Contents'].apply(self.filterString)
        logger.debug('Content filtered.')

    def dropEmptyContent(self):
        self.dataframe.dropna(subset=['Contents'], inplace=True)
        self.dataframe.drop(self.dataframe[self.dataframe['Contents'] == ''].index, inplace=True)
        logger.debug('Empty content dropped.')

    def dfToText(self):
        self.text = '\n'.join(self.dataframe.loc[:, 'Contents'].values)

    def generateVocab(self):
        self.vocab = sorted(set(self.text))
        logger.debug('Vocabulary generated.')
