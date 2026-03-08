from src.visualization.base import BaseVisualizer

# Import lazy: evita errori se matplotlib o pygame non sono installati
__all__ = ["BaseVisualizer", "MatplotlibVisualizer", "PygameVisualizer"]


def MatplotlibVisualizer(*args, **kwargs):
    from src.visualization.matplotlib_viz import MatplotlibVisualizer as _Cls
    return _Cls(*args, **kwargs)


def PygameVisualizer(*args, **kwargs):
    from src.visualization.pygame_viz import PygameVisualizer as _Cls
    return _Cls(*args, **kwargs)
