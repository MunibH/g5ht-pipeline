import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from matplotlib.gridspec import GridSpec
import matplotlib

def default_plt_params():
    font = {'family' : 'Arial',
            'weight' : 'normal',
            'size'   : 15}
    matplotlib.rc('font', **font)

def pretty_plot(figsize=(6,4), tick_dir='out', tick_length=5, tick_width=1, spine_width=0.75, fontsize=20, top_border=False, right_border=False):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.tick_params(direction=tick_dir, length=tick_length, width=tick_width)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    if not top_border:
        ax.spines['top'].set_visible(False)
    if not right_border:
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig, ax

def reflow_gen(fig):
    # we own this figure now so clear it
    fig.clf()
    # running count of the number of axes
    axcount = count(1)

    # the shape of the grid
    row_guess = col_guess = 0
    # the current GridSpec object
    gs = None

    # we are a generator, so loop forever
    while True:
        # what number is this Axes?
        j = next(axcount)
        # do we need to re-flow?
        if j > row_guess * col_guess:
            # Find the smallest square that will work
            col_guess = row_guess = int(np.ceil(np.sqrt(j)))
            # and then drop fully empty rows
            for k in range(1, row_guess):
                if (row_guess - 1) * col_guess < j:
                    break
                else:
                    row_guess -= 1

            # Create the new gridspec object
            gs = GridSpec(row_guess, col_guess, figure=fig)

            # for each of the axes, adjust it to use the new gridspec
            for n, ax in enumerate(fig.axes):
                ax.set_subplotspec(gs[*np.unravel_index(n, (row_guess, col_guess))])
            # resize the figure to have ~ 3:4 ratio and keep the Axes fixed
            fig.set_size_inches(col_guess * 4, row_guess * 3)

        # Add the new axes to the Figure at the next open space
        new_ax = fig.add_subplot(gs[*np.unravel_index(j - 1, (row_guess, col_guess))])

        # hand the Axes back to the user
        yield new_ax