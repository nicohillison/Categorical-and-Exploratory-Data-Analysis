'''data.py
Reads CSV files, stores data, access/filter data by variable name
Nico Hillison
CS 251 Data Analysis and Visualization
Spring 2021
'''
import csv
import numpy as np
import pandas as pd

class Data:
    def __init__(self, filepath=None, headers=None, data=None, header2col=None):
        '''Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each
            column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables
            (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in
                  as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0

        TODO:
        - Declare/initialize the following instance variables:
            - filepath
            - headers
            - data
            - header2col
            - Any others you find helpful in your implementation
        - If `filepath` isn't None, call the `read` method.
        '''
        self.filepath = filepath
        self.headers = headers
        self.data = data
        self.header2col = header2col

        if filepath != None:
            self.read(filepath)


    def read(self, filepath):
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called
        `self.data` at the end (think of this as 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
            NOTE: In the future, the Returns section will be omitted from docstrings if
            there should be nothing returned

        TODO:
        - Read in the .csv file `filepath` to set `self.data`. Parse the file to only store
        numeric columns of data in a 2D tabular format (ignore non-numeric ones). Make sure
        everything that you add is a float.
        - Represent `self.data` (after parsing your CSV file) as an numpy ndarray. To do this:
            - At the top of this file write: import numpy as np
            - Add this code before this method ends: self.data = np.array(self.data)
        - Be sure to fill in the fields: `self.headers`, `self.data`, `self.header2col`.
        

        NOTE: You may wish to leverage Python's built-in csv module. Check out the documentation here:
        https://docs.python.org/3/library/csv.html

        NOTE: In any CS251 project, you are welcome to create as many helper methods as you'd like.
        The crucial thing is to make sure that the provided method signatures work as advertised.

        NOTE: You should only use the basic Python library to do your parsing.
        (i.e. no Numpy or imports other than csv).
        Points will be taken off otherwise.

        TIPS:
        - If you're unsure of the data format, open up one of the provided CSV files in a text editor
        or check the project website for some guidelines.
        - Check out the test scripts for the desired outputs.
        '''
        self.filepath = filepath
        self.headers = []
        self.header2col={}
        self.data = []
        with open(filepath, 'r') as file:
            csv_reader = csv.reader(file, delimiter = ',')
            line1 = next(csv_reader)#should be header
            line2 = next(csv_reader)#should be datatype
            #check whether the line2 is a datatype row
            #what line 2 is it's a list of elements 
            for element in line2:
                try:
                    a = float(element)
                    raise Exception('')
                except ValueError:
                    continue
            #try except
            line3 = next(csv_reader)#should be the start of the actual data
            #print(line3)
            #try except 
            r = 0
            a = 0
            numlist = []
            for element in line3:
                try:
                    numlist.append(float(element))
                    self.headers.append(line1[r])
                    self.header2col[line1[r].strip()]=a
                    a+=1
                except ValueError:
                    pass
                r+=1
            self.data.append(numlist)
            for row in csv_reader:
                #print(row)
                #break
                numlist = []
                for val in row:
                    try: 
                        numlist.append(float(val))
                    except ValueError:
                        continue
                self.data.append(numlist)
                # print(type(row[0]))
                # if row[0].isnumeric():
                #     self.data.append(row) 
        #print(self.data)
        self.data = np.array(self.data)




    def __str__(self):
        '''toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's
        called to determine what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''
        s =  'the first 5 rows of data are: \n' + str(self.data[0:5, : ])

        return s


    def get_headers(self):
        '''Get method for headers

        Returns:
        -----------
        Python list of str.
        '''
        # self.headers = []
        # for row in self.data:
        #     if row.isinstance(object, str) = True:
        #         self.headers.append(row)

        # pass
        return self.headers

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        return self.header2col

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        return np.size(self.data)
        #print(len(self.data))
        #print(np.size(self.data))
        #print(self.data)
        #print(type(self.data))
        #return self.data.ndim()
        #return self.data.shape[1]

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        return len(self.data)

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''
        return self.data[rowInd]

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''
        idx_list = []
        for h in headers:
            idx_list.append(self.header2col[h])
        return idx_list

    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself.
            This can be accomplished with numpy's copy function.
        '''
        return np.copy(self.data)

    def head(self):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''
        return self.data[0:5, : ]

    def tail(self):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''
        return self.data[-5:, : ]

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        '''
        self.data = self.data[start_row : end_row, : ]
        return self.data


    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified
        by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return
        column #2 of self.data. If rows is not [] (say =[0, 2, 5]), then we do the same thing,
        but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select.
                Empty list [] means take all rows

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''


    
        header_num = []

        for h in headers:
            if h in self.header2col:
                header_num.append(self.header2col[h])
        
        #print(type(self.data))

        if rows == []:
            return self.data[ : , header_num]
        else:
            return self.data[np.ix_(rows, header_num)]

        
        

