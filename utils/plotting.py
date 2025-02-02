from collections.abc import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def subplots(
    names: tuple[str, ...],
    data_generator: Callable[[str], pd.DataFrame | pd.Series],
    plot_generator: Callable[[str, pd.DataFrame | pd.Series, Axes], None],
    title_generator: Callable[[str, pd.DataFrame | pd.Series], str] = lambda name, data: name,
    n_cols: int = 5, font_size: int = 10,
    x_label: str = None, y_label: str = None,
) -> None:
    plt.rcParams.update({'font.size': font_size})

    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 3.5, n_rows * 3))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)

    for ax, name in zip(axs.flatten(), names):
        data = data_generator(name)
        plot_generator(name, data, ax)
        ax.set_title(title_generator(name, data))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    plt.show()
