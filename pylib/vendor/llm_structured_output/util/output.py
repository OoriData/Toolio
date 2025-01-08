# pylint: disable=missing-function-docstring
"""
Terminal colored output
"""


def info(*args, **kwargs):
    print("\033[34mℹ ", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def warning(*args, **kwargs):
    print("\033[43;37m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def debug(*args, **kwargs):
    print("\033[33m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def debugbold(*args, **kwargs):
    print("\033[1;33m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def bold(*args, **kwargs):
    print("\033[1;30m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def bolddim(*args, **kwargs):
    print("\033[1;2;30m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def boldalt(*args, **kwargs):
    print("\033[1;36m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def underline(*args, **kwargs):
    print("\033[4m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def inverse(*args, **kwargs):
    print("\033[7m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def setfg(r: float, g: float, b: float):
    """Each of r,g,b must be between 0 and 1"""
    color = 16 + 36 * round(5 * r) + 6 * round(5 * g) + round(5 * b)
    print(f"\033[38;5;{color}m", end="")


def setbg(r: float, g: float, b: float):
    """Each of r,g,b must be between 0 and 1"""
    color = 16 + 36 * round(5 * r) + 6 * round(5 * g) + round(5 * b)
    print(f"\033[48;5;{color}m", end="")


def clear():
    print("\033[0m", end="")
