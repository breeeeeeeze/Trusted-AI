import datetime


def convertSnowflake(snowflake: int) -> datetime.datetime:
    """
    Converts a discord snowflake to a datetime object
    """
    converted = datetime.datetime.fromtimestamp(((snowflake >> 22) + 1420070400000)/1000, datetime.timezone.utc)
    return converted
