import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from matplotlib.gridspec import GridSpec


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