#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     util_tute04.py                                                    #
#                                                                             #
# PURPOSE:  Utility code for AST3110 Tutorial 4 at Macquarie University.      #
#                                                                             #
# MODIFIED: 01-Apr-2020 by C. Purcell                                         #
#                                                                             #
#=============================================================================#
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


#-----------------------------------------------------------------------------#
def poly5(p):
    """
    When called, this function takes a vector of parameters
    and returns another functions to evaluate a polynomial
    with these coefficients fixed.
    """

    # Pad out p vectors smaller than len = 6 (longhand version)
    #numPad = 6 - len(p)
    #pad = np.zeros((numPad))
    #p = np.append(pad, p)

    # Pad out p vectors smaller than len = 6 (shorthand version)
    p = np.append(np.zeros((6-len(p))), p)

    def rfunc(x):
        """
        This function is returned by the poly5 function. It takes a
        vector of x values and evaluated the current polynomial.
        """
        y = (p[0]*x**5.0 + p[1]*x**4.0 + p[2]*x**3.0 +
             p[3]*x**2.0 + p[4]*x + p[5])

        # Note the indent here
        return y

    # Note the indent here
    return rfunc


#-----------------------------------------------------------------------------#
def plot_spec_poly5(xData, yData, dyData, p=None):
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
        yModel = poly5(p)(xModel)

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
