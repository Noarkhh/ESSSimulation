import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.offsetbox import AnchoredText
import numpy as np
matplotlib.use('TkAgg')


beh_names = ["doves", "hawks", "retaliators", "bullies", "probers"]
beh_colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])


def plot_data(population):
    gridsize = (10, 14)
    fig = plt.figure(figsize=(10, 8))

    ax0 = plt.subplot2grid(gridsize, (0, 0), rowspan=4, colspan=14)
    ax0.set_title("Behaviors in the population over time")
    for beh_id in population.behs:
        ax0.plot(population.history[beh_id], label=beh_names[beh_id], color=beh_colors[beh_id])
    ax0.set_ylim([0, population.size])
    ax0.set_ylabel("amount")
    ax0.set_xlabel("generation")
    anchored_text = AnchoredText(f"Population size: {population.size}     "
                                 f"Generations: {population.generation}\n"
                                 f"Fitness offspring factor: {population.fitness_offspring_factor}  "
                                 f"Random offspring factor: {population.random_offspring_factor}", loc="upper center")
    anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

    ax0.add_artist(anchored_text)
    ax0.legend(loc="upper right")

    ax1 = plt.subplot2grid(gridsize, (6, 0), rowspan=4, colspan=6)
    ax1.set_title("Distribution of behaviors in the starting population")
    rects1 = ax1.bar([beh_names[i] for i in population.behs],
                     [population.history[beh_id][0] / population.size for beh_id in population.behs],
                     color=beh_colors[population.behs])
    ax1.bar_label(rects1,
                  labels=[str(round(population.history[beh_id][0] / population.size * 100, 2)) + '%' for beh_id in
                          population.behs], fmt='%1.2f%%', padding=3)
    ax1.set_ylim([0, 1.1])
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))

    ax2 = plt.subplot2grid(gridsize, (6, 8), rowspan=4, colspan=6)
    ax2.set_title("Distribution of behaviors in the final population")
    rects2 = ax2.bar([beh_names[i] for i in population.behs],
                     [population.history[beh_id][-1] / population.size for beh_id in population.behs],
                     color=beh_colors[population.behs])
    ax2.bar_label(rects2,
                  labels=[str(round(population.history[beh_id][-1] / population.size * 100, 2)) + '%' for beh_id in
                          population.behs], padding=3)
    ax2.set_ylim([0, 1.1])
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
    plt.subplots_adjust(left=0.1, bottom=0.06, right=0.95, top=0.95, wspace=0, hspace=0)
    plt.show()
