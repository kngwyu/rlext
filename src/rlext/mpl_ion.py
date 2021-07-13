import matplotlib as mpl


class ModeHolder:
    def __init__(self):
        self._inner = False

    def __bool__(self):
        return self._inner


JUPYTER_MODE = ModeHolder()


def jupyter_mode(mode=True):
    global JUPYTER_MODE
    JUPYTER_MODE._inner = True


def nogui_mode():
    from matplotlib import pyplot as plt

    mpl.use("agg")
    plt.ioff()

    def _stub(*args, **kwargs):
        pass

    plt.show = _stub


def __mpl_select_backend() -> None:
    backend = mpl.get_backend().lower()
    if any(map(lambda gui: backend.startswith(gui), ["qt", "tk", "gtk"])):
        return
    else:
        mpl.use("TkAgg")


# Try GUI backend first
try:
    __mpl_select_backend()
    from matplotlib import pyplot as plt

    plt.ion()
except ImportError:
    from matplotlib import pyplot as plt

    nogui_mode()
