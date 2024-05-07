'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable as pa
import analysis
import data


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''
        super().__init__(orig_dataset)
        #print(orig_dataset.get_num_dims())
        self.orig_dataset = orig_dataset
        #print(self.orig_dataset.get_num_dims())
        self.data = data

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''
        #print(self.orig_dataset.get_num_dims())
        dataArray = self.orig_dataset.select_data(headers)
        # dataArray = self.orig_dataset.get_all_data()
        header2col = dict()
        for h in range(len(headers)):
            header2col[headers[h]] = h
        self.data = data.Data(headers=headers, data=dataArray, header2col=header2col)
        #print(self.orig_dataset.get_num_dims())

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        # get a copy of the projected data - stored in self.data
        proj_data = self.data.get_all_data()
    
        # cant get shape
        # print(type(proj_data))
        # print(proj_data.shape)
        # print(proj_data.data[0])
        ones = np.ones([proj_data.shape[0], 1])

        homog_cord = np.hstack([proj_data, ones])

        return homog_cord

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        #print('orig dataset shape:', self.orig_dataset.shape)
        #print('data:', self.data)
        M = len(magnitudes)


        identity = np.eye(M, M)
        magnitudes = np.expand_dims(np.array(magnitudes), axis = 1)
        #print(len(magnitudes))
        temp = np.hstack((identity, magnitudes))
        lst = [0] * M + [1]
        final = np.vstack((temp, lst))

        return final

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        #M = self.get_num_dims()
        M = len(magnitudes)

        scale = np.eye(M+1, M+1)
        for i in range(M):
            scale[i,i] = magnitudes[i]

        return scale


    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        proj_data = self.get_data_homogeneous()
        #print('projected_data shape: ', proj_data.shape)

        translation_matrix = self.translation_matrix(magnitudes)
        #print('translation_matrix shape: ', translation_matrix)
        #print('translation matrix:', translation_matrix.shape)

        translation = (proj_data @ translation_matrix.T)
        less = np.delete(translation, -1, 1)
        #print('translation shape: ', translation.shape)

        self.data = data.Data(headers=self.data.get_headers(), data=less, header2col=self.data.get_mappings())

        return self.data

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        proj_data = self.get_data_homogeneous()
        #print('projected_data shape: ', proj_data.shape)

        scale_matrix = self.scale_matrix(magnitudes)
        #print('scale_matrix shape: ', scale_matrix.shape)
        #print('translation matrix:', translation_matrix.shape)


        scaling = (proj_data @ scale_matrix.T)
        less = np.delete(scaling, -1, 1)
        #print('translation shape: ', translation.shape)

        self.data = data.Data(headers=self.data.get_headers(), data=less, header2col=self.data.get_mappings())

        return self.data


    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        proj_data = self.get_data_homogeneous()
        #print('projected_data shape: ', proj_data.shape)

        #scale_matrix = self.scale_matrix(magnitudes)
        #print('scale_matrix shape: ', scale_matrix.shape)
        #print('transformation shape: ', transformation.shape)


        transformation = (proj_data @ C.T)
        less = np.delete(transformation, -1, 1)
        #print('transformation shape: ', transformation.shape)

        self.data = data.Data(headers=self.data.get_headers(), data=less, header2col=self.data.get_mappings())

        return less

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        mins = np.amin(self.min(self.data.get_headers()), 0)
        maxs = np.amax(self.max(self.data.get_headers()), 0)

        lst = [-1 * mins] * (self.data.data.shape[1])
        trans_m = self.translation_matrix(lst)

        lst2 = [1/(maxs-mins)] * (self.data.data.shape[1])
        scale_m = self.scale_matrix(lst2)

        C = scale_m @ trans_m

        return self.transform(C)

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        mins = self.min(self.data.get_headers())
        maxs = self.max(self.data.get_headers())

        
        trans_m = self.translation_matrix(-1 * mins)

        scale_m = self.scale_matrix(1/(maxs-mins))

        C = scale_m @ trans_m

        trans_d = self.transform(C)
        #print('this is trans_data: ' , type(trans_d))

        self.data = data.Data(headers=self.data.get_headers(), data=trans_d, header2col=self.data.get_mappings())
        #print(type(self.data))

        return trans_d

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        h = self.data.get_header_indices([header])
        #print(h)

        id = np.eye(self.data.data.shape[1]+1)

        if h[0] == 0:
            id[1:3,1:3] = [[np.cos(np.radians(degrees)), -np.sin(np.radians(degrees))], [np.sin(np.radians(degrees)), np.cos(np.radians(degrees))]]
        elif h[0] == 1:
            id[0,0] = np.cos(np.radians(degrees))
            id[0,2] = np.sin(np.radians(degrees))
            id[2,0] = -np.sin(np.radians(degrees))
            id[2,2] = np.cos(np.radians(degrees))
        else: 
            id[0:2,0:2] = [[np.cos(np.radians(degrees)),-np.sin(np.radians(degrees))], [np.sin(np.radians(degrees)), np.cos(np.radians(degrees))]]
        
        return id

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        proj_data = self.get_data_homogeneous()
        print('projected_data shape: ', proj_data.shape)

        rotation_matrix = self.rotation_matrix_3d(header, degrees)
        #print('scale_matrix shape: ', scale_matrix.shape)
        print('rotation matrix:', rotation_matrix.shape)


        rotating = (proj_data @ rotation_matrix.T)
        less = np.delete(rotating, -1, 1)
        #print('translation shape: ', translation.shape)

        self.data = data.Data(headers=self.data.get_headers(), data=less, header2col=self.data.get_mappings())

        return self.data

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        plt.figure(figsize = (6,6))
        plt.scatter(self.data.select_data([ind_var]), self.data.select_data([dep_var]), c = self.data.select_data([c_var]), cmap = pa.colorbrewer.sequential.Greys_7.mpl_colormap)
        #c = self.data.select_data[c_var]
        #cmap = pa.colorbrewer.qualitative.Accent_4
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        #plt.clabel(c_var)
        # cbar = plt.colorbar()
        plt.colorbar(label = c_var)


        


    def heatmap(self, headers=None, title=None, cmap="gray"):
        '''Generates a heatmap of the specified variables (defaults to all). Each variable is normalized
        separately and represented as its own row. Each individual is represented as its own column.
        Normalizing each variable separately means that one color axis can be used to represent all
        variables, 0.0 to 1.0.

        Parameters:
        -----------
        headers: Python list of str (or None). (Optional) The variables to include in the heatmap.
            Defaults to all variables if no list provided.
        title: str. (Optional) The figure title. Defaults to an empty string (no title will be displayed).
        cmap: str. The colormap string to apply to the heatmap. Defaults to grayscale
            -- black (0.0) to white (1.0)

        Returns:
        -----------
        fig, ax: references to the figure and axes on which the heatmap has been plotted
        '''

        # Create a doppelganger of this Transformation object so that self.data
        # remains unmodified when heatmap is done
        data_clone = data.Data(headers=self.data.get_headers(),
                               data=self.data.get_all_data(),
                               header2col=self.data.get_mappings())
        dopp = Transformation(self.data, data_clone)
        dopp.normalize_separately()

        fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.set(xlabel="Individuals")

        # Select features to plot
        if headers is None:
            headers = dopp.data.headers
        m = dopp.data.select_data(headers)

        # Generate heatmap
        hmap = ax.imshow(m.T, aspect="auto", cmap=cmap, interpolation='None')

        # Label the features (rows) along the Y axis
        y_lbl_coords = np.arange(m.shape[1]+1) - 0.5
        ax.set_yticks(y_lbl_coords, minor=True)
        y_lbls = [""] + headers
        ax.set_yticklabels(y_lbls)
        ax.grid(linestyle='none')

        # Create and label the colorbar
        cbar = fig.colorbar(hmap)
        cbar.ax.set_ylabel("Normalized Features")

        return fig, ax
