# toolio.tool.demo

from toolio.tool import tool, param


@tool('birthday_lookup',
      desc='Look up employeess by birthday, plus some information on their interests',
      params=[param('date', str, 'date to check for employees. Must be in the form `MM-DD`, e.g. `05-03` for May 3rd.', True)])
def birthday_lookup(date=None):
    '''
    Demo of value look up from a table. Same approach could be used to query a DBMS as well.
    '''
    BIRTHDAY_DB = {
        '07-01': [('Obi', 'enjoys gardening and karaoke')],
        '10-25': [('Ada', 'collects afrobeat records and coaches soccer')],
        }
    if date in BIRTHDAY_DB:
        birthday_info = BIRTHDAY_DB[date]
        birthday_text = [f'* {i[0]} who {i[1]}\n' for i in birthday_info]
        return f'It\'s a birthday for: {birthday_text}'
    else:
        return 'No one has a birthday today'


@tool('today',
      desc='Get the current date')
def today_kfabe(key=None):
    '''
    For demo purposes always give a pretend date for today
    '''
    return '07-01'
    # return 'Monday, July 1, 2024'
