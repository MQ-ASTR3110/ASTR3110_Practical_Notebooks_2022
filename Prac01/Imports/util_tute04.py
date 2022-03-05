#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     util_tute04.py                                                    #
#                                                                             #
# PURPOSE:  Utility code for AST3110 Tutorial 4 at Macquarie University.      #
#                                                                             #
# MODIFIED: 01-Apr-2020 by C. Purcell, 17-Mar-2021 by M.Owers                 #
#                                                                             #
#=============================================================================#
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


#-----------------------------------------------------------------------------#

def polyN(p):
    """
    When called, this function takes a vector of parameters
    and returns another function to evaluate a polynomial
    with these coefficients fixed.
    """
    
    def rfunc(x):
        """
        This function is returned by the polyN function. It takes a
        vector of x values and loops over the order number, generating an Nth order polynomial.
        """
        y = 0
        for i in np.arange(len(p)):
            y += p[i]*x**float(i)
        # Note the indent here
        return y

    # Note the indent here
    return rfunc

#-----------------------------------------------------------------------------#
def plot_spec_polyN(xData, yData, dyData, p=None):
    """
    Function to plot a spectrum and (optionally) a model polynomial fit.
    """

    # Setup the figure
    fig = plt.figure()
    fig.set_size_inches([12,12])
    ax = fig.add_subplot(1,1,1)

    # First plot the data
    ax.errorbar(x=xData, y=yData, yerr=dyData, mfc="none",
                ms=4, fmt="D", ecolor="grey", label="Data",
                elinewidth=1.0, capsize=2)

    # Only plot the model curve if p has been specified
    if p is not None:

        # Make a model curve, sampled at small
        # intervals to appear smooth
        nSamples = 100
        xModel = np.linspace(start=np.min(xData),
                             stop=np.max(xData),
                             num=nSamples)
        yModel = polyN(p)(xModel)

        # Plot the model
        ax.plot(xModel, yModel, color="red", marker="None",
                mfc="w", mec="g", label="Model", lw=2.0)

    # Set the labels
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Amplitude (mJy)')


#-----------------------------------------------------------------------------#
def plot_trace(sampler, figSize=(12, 12)):

    # Parse the shape of the sampler array
    nWalkers, nSteps, nDim = sampler.chain.shape

    # Initialise the figure
    fig = plt.figure(figsize=figSize)

    # Plot a trace for each parameter
    for j in range(nDim):

        # Extract the arrays we want to plot
        chain = sampler.chain[:,:,j].transpose()
        like = sampler.lnprobability.transpose()
        ax = fig.add_subplot(nDim, 1, j+1)
        stepArr = np.arange(nSteps, dtype="f4") + 1

        # Loop through the walkers
        for i in range(nWalkers):
            ax.scatter(x=stepArr, y=chain[:,i], c=like[:,i],
                       cmap=plt.cm.jet, marker="D", edgecolor='none',
                       alpha=0.2, s=4)
            ax.set_ylabel("P {:d}".format(j + 1))
            if j < nDim - 1:
                [label.set_visible(False) for label in ax.get_xticklabels()]

        # Format the axes
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        yRange = np.max(chain) - np.min(chain)
        yMinPlt = np.min(chain) - yRange*0.1
        yMaxPlt = np.max(chain) + yRange*0.1
        ax.set_ylim(yMinPlt, yMaxPlt)
        ax.set_xlim(np.min(stepArr)- nSteps*0.01, np.max(stepArr)
                    + nSteps*0.01)

    # Label the x axis and format the spacing
    ax.set_xlabel('Steps', fontsize = 15)
    fig.subplots_adjust(left=0.18, bottom=0.07, right=0.97, top=0.94)
