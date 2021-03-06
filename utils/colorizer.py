class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    MAGENTA = '\033[35m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    RED = '\033[31m'
    BACKGROUND_WHITE = '\033[47m'


def colorize(text: str, *colors: str) -> str:
    """
    Colorize the given text for terminal output.
    """
    colorString = ''.join(getattr(bcolors, color) for color in colors)
    return f'{colorString}{text}{bcolors.ENDC}'
