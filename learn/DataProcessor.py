import glob
import logging
from typing import Tuple, List, Optional

import regex as re
import pandas as pd

from utils.configReader import readConfig
from utils.colorizer import colorize

config = readConfig()
logger = logging.getLogger('ai.learn.dataprocessor')


class DataProcessor:
    """
    Class to import and process training data.
    """
    def __init__(self) -> None:
        self.dataframe = pd.DataFrame()
        self.text: Optional[str] = None
        self.vocab = []

    def processInputData(self) -> Tuple[str, List[str]]:
        """
        Import and process data, returns text and vocabulary.
        """
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
        return self.text, self.vocab  # type: ignore

    def slimTo(self, count: int) -> None:
        """
        Slim the dataframe to n rows. Useful for debugging.

        :param int count: Number of rows to keep.
        """
        self.dataframe = self.dataframe[:count]

    def resetIndex(self) -> None:
        """
        Reset index of the dataframe.
        """
        self.dataframe.reset_index(drop=True, inplace=True)
        logger.debug('Reindexed.')

    def getVocab(self) -> List[str]:
        """
        Get the vocabulary.

        :return: Vocabulary.
        """
        if self.vocab:
            return self.vocab
        self.processInputData()
        return self.vocab

    def exportVocab(self, fileName: str) -> None:
        """
        Export vocabulary to file.

        :param str fileName: Filename.
        """
        with open(fileName, 'w', encoding='utf-8') as f:
            for word in self.vocab:
                f.write(word)
        logger.debug('Vocabulary exported.')

    def importVocab(self, fileName: str) -> List[str]:
        """
        Import vocabulary from file.

        :param str fileName: Filename.
        :return: Vocabulary.
        """
        with open(fileName, 'r', encoding='utf-8') as f:
            self.vocab = sorted(set(f.read()))
        logger.debug('Vocabulary imported.')
        return self.vocab

    def sortByTimestamp(self) -> None:
        """
        Sort the dataframe by timestamp.
        """
        if 'Timestamp' not in self.dataframe.columns:
            logger.error(
                colorize(
                    'Failed to sort dataframe. Timestamp not found in dataframe.',
                    'FAIL',
                )
            )
        self.dataframe.sort_values('Timestamp', inplace=True)  # type: ignore
        self.dataframe.drop('Timestamp', axis=1, inplace=True)
        logger.debug('Sorted by timestamp')

    def importData(self, fileName: str = '') -> None:
        """
        Import data from specified file or from globs specified in config (default).
        """
        listOfDFs = []
        if fileName:
            listOfDFs.append(pd.read_csv(fileName))
        else:
            for fileGlob in config['learn']['data']['inputFiles']:
                for fileName in glob.glob(f'data\\{fileGlob}'):
                    logger.debug(f'Importing {fileName}')
                    listOfDFs.append(pd.read_csv(fileName))
        self.dataframe = pd.concat(listOfDFs)
        if 'Contents' not in self.dataframe.columns:
            raise Exception('Contents column not found in dataframe.')
        logger.debug('Data imported.')

    def dropColumns(self) -> None:
        """
        Drop all unnecessary columns.
        """
        if 'ChannelID' in self.dataframe.columns:
            self.dataframe.drop('ChannelID', axis=1, inplace=True)
        if 'AuthorID' in self.dataframe.columns:
            self.dataframe.drop('AuthorID', axis=1, inplace=True)
        if 'ID' in self.dataframe.columns:
            self.dataframe.drop('ID', axis=1, inplace=True)
        if 'Timestamp' in self.dataframe.columns:
            self.dataframe.drop('Timestamp', axis=1, inplace=True)
        logger.debug('Columns dropped.')

    def dropAttachments(self) -> None:
        """
        Drop rows that contain attachments, if enabled in config. Then drop the attachments column.
        """
        if config['learn']['data']['removeRowIfAttachment']:
            self.dataframe = self.dataframe[self.dataframe['Attachments'].isnull()]
        if 'Attachments' in self.dataframe.columns:
            self.dataframe.drop('Attachments', axis=1, inplace=True)
        logger.debug('Attachments dropped.')

    @staticmethod
    def filterString(string: str) -> str:
        """
        Filter the string based on settings specified in config.

        :param str string: String to filter.
        """
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
        # reverse Discords way of escaping quotes in the content field
        string = re.sub(r'""', '"', string)
        return string

    def filterContent(self) -> None:
        """
        Apply the filter to the entire dataframe.
        """
        self.dataframe['Contents'] = self.dataframe['Contents'].apply(self.filterString)
        logger.debug('Content filtered.')

    def dropEmptyContent(self) -> None:
        """
        Drop all rows that are empty.
        """
        self.dataframe.dropna(subset=['Contents'], inplace=True)  # type: ignore
        self.dataframe.drop(self.dataframe[self.dataframe['Contents'] == ''].index, inplace=True)
        logger.debug('Empty content dropped.')

    def dfToText(self) -> None:
        """
        Convert the dataframe to a single string.
        """
        self.text = '\n'.join(self.dataframe.loc[:, 'Contents'].values)

    def generateVocab(self) -> None:
        """
        Generate the vocabulary based on the text string
        """
        self.vocab = sorted(set(self.text))  # type: ignore
        logger.debug('Vocabulary generated.')
