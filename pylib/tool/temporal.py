# toolio.tool.temporal

import datetime

from toolio.tool import tool, param


@tool('current_time', params=[])
def current_time():
    '''
    Get the current date and time
    '''
    t = datetime.datetime.today()
    return t.ctime()
