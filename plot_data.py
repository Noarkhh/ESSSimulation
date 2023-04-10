from __future__ import annotations
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulation import Population
# matplotlib.use('TkAgg')

# plt.style.use('dark_background')

behavior_names = ["doves", "hawks", "retaliators", "bullies", "probers"]
beh_colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])


def plot_population(population: Population, description: str = "") -> None:
    df: pd.DataFrame = population.history_to_dataframe()
    gridsize = (20, 14)
    fig = plt.figure(figsize=(9, 7))
    fig.patch.set_facecolor('#8e9299')

    ax0 = plt.subplot2grid(gridsize, (0, 0), rowspan=8, colspan=14)
    df.plot(ax=ax0, color=beh_colors[population.behaviors], ylabel="amount", xlabel="generation")

    ax0.set_title("Behaviors in the population over time")
    ax0.set_ylim([0, population.size])
    anchored_text = AnchoredText(f"Population size: {population.size}     "
                                 f"Generations: {population.generation}\n"
                                 f"Fitness offspring factor: {population.fitness_offspring_factor}  "
                                 f"Random offspring factor: {population.random_offspring_factor}", loc="upper center")
    anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

    ax0.add_artist(anchored_text)
    ax0.legend(loc="upper right")
    ax0.patch.set_facecolor('#ccd0d8')

    ax1 = plt.subplot2grid(gridsize, (11, 0), rowspan=8, colspan=6)
    ax1.set_title("Distribution of behaviors in the starting population")
    rects1 = ax1.bar(df.keys(), np.array(df.head(1))[0] / population.size, color=beh_colors[population.behaviors])
    ax1.bar_label(rects1,
                  labels=[f"{number:.2f}%" for number in np.array(df.head(1))[0] / population.size * 100], padding=3)
    ax1.set_ylim([0, 1.1])
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
    ax1.patch.set_facecolor('#ccd0d8')

    ax2 = plt.subplot2grid(gridsize, (11, 8), rowspan=8, colspan=6)
    ax2.set_title("Distribution of behaviors in the final population")
    rects2 = ax2.bar(df.keys(), np.array(df.tail(1))[0] / population.size, color=beh_colors[population.behaviors])
    ax2.bar_label(rects2,
                  labels=[f"{number:.2f}%" for number in np.array(df.tail(1))[0] / population.size * 100], padding=3)
    ax2.set_ylim([0, 1.1])
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
    ax2.patch.set_facecolor('#ccd0d8')

    plt.subplots_adjust(left=0.1, bottom=0.06, right=0.95, top=0.95, wspace=0, hspace=0)
    plt.figtext(0.5, 0.0, description, wrap=True, horizontalalignment='center', verticalalignment='top', fontsize=12)
    plt.show()


def plot_ratios(df: pd.DataFrame, behaviors: tuple[int, int] = (0, 1), description: str = "") -> None:
    fig = plt.figure()
    fig.patch.set_facecolor('#8e9299')

    ax = fig.add_subplot(111)
    df.plot(ax=ax, color=beh_colors[np.array(behaviors)],
            xlabel=f"percentage of {df.keys()[0]} in the population",
            ylabel="expected encounter reward",
            title="Expected encounter results by\nbehavior ratios in the population")
    ax.patch.set_facecolor('#ccd0d8')

    plt.figtext(0.5, -0.07, description, wrap=True, horizontalalignment='center', verticalalignment='top', fontsize=12)
    plt.show()
