# toolio.tool.math

import re2 as re

from toolio.tool import tool, param

# Restricted to numbers & arithmetic symbols (& e for exponents, which does open up an infinitesimal attack vector)
ALLOWED_EXPR_PAT = re.compile(r'[\d\.eE\+\-\*\/^\(\)]+')

# Note: One could implement a more complete calculator such as with SymPy, but there are security implications of that
# Since it uses eval() under the hood, and it's a terrible idea to eval from sources that are not carefully controlled
# See, for example: https://github.com/sympy/sympy/issues/10805
# In general, see: https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
# We just keep this super simple and restricted for those safety reasons
@tool('calculator', params=[param('expr', str, 'mathematical expression to be computed', True)])
def calculator(expr=None):
    '''
    Make an arithmetical, mathematical calculation, using operations such as addition (+), subtraction (-),
    multiplication (*), and division (/). Don't forget to use parenthesis for grouping.
    **Always use this tool for calculations. Never try to do them yourself**.
    '''
    # print(repr(expr))
    if not ALLOWED_EXPR_PAT.match(expr):
        raise ValueError(f'Disallowed characters encountered in mathematical expression {expr}')
    result = eval(expr, {})
    return result
