'''analysis.py
Run statistical analyses and plot Numpy ndarray data
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from data import Data


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        # header_indice = data.get_header_indices(headers)
        return np.min(self.data.select_data(headers, rows), axis = 0)
        

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        return np.max(self.data.select_data(headers, rows), axis = 0)

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        #return np.subtract(self.max(headers, rows), self.min(headers, rows))
        return (self.min(headers, rows), (self.max(headers, rows)))

    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: Loops are forbidden!
        '''
        return np.sum(self.data.select_data(headers, rows), 0) / len(self.data.select_data(headers, rows))

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: Loops are forbidden!
        '''
        return np.sum(((self.data.select_data(headers, rows) - self.mean(headers, rows))**2), axis = 0) / (len(self.data.select_data(headers, rows)) - 1)

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: Loops are forbidden!
        '''
        return np.sqrt(self.var(headers, rows))

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        plt.figure(figsize = (7,7))
        plt.scatter(self.data.select_data([ind_var]), self.data.select_data([dep_var]))
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        


        return self.data.select_data([ind_var]).flatten(), self.data.select_data([dep_var]).flatten()

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''

        fig, axes = plt.subplots(len(data_vars),len(data_vars), sharex= 'col', sharey= 'row', figsize = fig_sz)
        fig.suptitle(title)
        counter = 0
        col = 0
        row = 0
        for col in range(len(data_vars)):
            for row in range(len(data_vars)):
                counter = counter + 1
                # p = plt.subplot(len(data_vars), len(data_vars), counter)
                axes[col, row].scatter(self.data.select_data([data_vars[row]]), self.data.select_data([data_vars[col]]))
                # plt.scatter(self.data.select_data([data_vars[row]]), self.data.select_data([data_vars[col]]))
                # plt.tick_params(labelcolor='none', top=False, bottom=True, left=True, right=False)
               #creates x-axis only on the bottom row and y-axis on the first column 
                if row == 0:
                    # plt.ylabel(data_vars[col])
                    axes[col, row].set_ylabel(data_vars[col])
                
                if col == len(data_vars)-1:
                    axes[col, row].set_xlabel(data_vars[row])
                
      
        #first column
        #axes[3,0].scatter(self.data.select_data(['sepal_length']), self.data.select_data(['sepal_width']))
        #axes[2,0].scatter(self.data.select_data(['sepal_length']), self.data.select_data(['petal_length']))
        #axes[1,0].scatter(self.data.select_data(['sepal_length']), self.data.select_data(['petal_width']))
        #axes[0,0].scatter(self.data.select_data(['sepal_length']), self.data.select_data(['sepal_length']))
        #second column
        #axes[3,1].scatter(self.data.select_data(['sepal_width']), self.data.select_data(['sepal_width']))
        #axes[2,1].scatter(self.data.select_data(['sepal_width']), self.data.select_data(['petal_length']))
        #axes[1,1].scatter(self.data.select_data(['sepal_width']), self.data.select_data(['petal_width']))
        #axes[0,1].scatter(self.data.select_data(['sepal_width']), self.data.select_data(['sepal_length']))
        #third column
        #axes[3,2].scatter(self.data.select_data(['petal_length']), self.data.select_data(['sepal_width']))
        #axes[2,2].scatter(self.data.select_data(['petal_length']), self.data.select_data(['petal_length']))
        #axes[1,2].scatter(self.data.select_data(['petal_length']), self.data.select_data(['petal_width']))
        #axes[0,2].scatter(self.data.select_data(['petal_length']), self.data.select_data(['sepal_length']))
        #fourth column
        #axes[3,3].scatter(self.data.select_data(['petal_width']), self.data.select_data(['sepal_width']))
        #axes[2,3].scatter(self.data.select_data(['petal_width']), self.data.select_data(['petal_length']))
        #axes[1,3].scatter(self.data.select_data(['petal_width']), self.data.select_data(['petal_width']))
        #axes[0,3].scatter(self.data.select_data(['petal_width']), self.data.select_data(['sepal_length']))

    

        return fig, axes
        




    


