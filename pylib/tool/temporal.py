# SPDX-FileCopyrightText: 2024-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
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
