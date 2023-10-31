"""
**The `TFData` class declaration.**
"""
import numpy as np
import xarray as xr

class TFData:
    ''' An object based on the `xarray.DataArray` class for storing TFD results.
    The approach is to use a composition rather than subclassing (inheriting) from
    the `DataArray` because inheritance with xarray is shaky and unstable, as the
    authors admit themselves - see
    https://docs.xarray.dev/en/stable/internals/extending-xarray.html.

    `TFData` contains a 3-dimensional `DataArray` with shape nchans x nfreqs x nparms as a
    `data` attribute. `data` has dimensions (axes) labeled as `'chan', 'frq', 'parm'`
    respectively. Labels (ticks) along each dimension (the 'coords' in the xarray
    terminology) are given by lists `ch_names`, `freqs` and `parm_names`, respectively.
    If not supplied, integer indices along each dimension will be used.

    **Attributes**

    Attributes:
        CHAN_DIM (str): a class constant for the xarray "channels" dimension; set to "chan"
        FRQ_DIM (str):  a class constant for the xarray "frequency" dimension; set to "frq"
        PARM_DIM (str):  a class constant for the xarray "parameters" dimension; set to "parm"
        dim_names (tuple of str): a class attribute. Contains symbolic names of the
            3 xarray dimensions `CHAN_DIM, FRQ_DIM, PARM_DIM`
        scan_id(str or None): the ID of the input record.
        ch_names (list of str or None): labels along the `CHAN_DIM` dimension.
        freqs (list of floats or None): labels along the `FRQ_DIM` dimension.
        parm_names (list of str or None): labels along the `PARM_DIM` dimension
        data (xarray): the internal xarray object that stores the data
            
    **Methods**

    '''
    # Dimension name constants
    CHAN_DIM = 'chan'
    FRQ_DIM = 'frq'
    PARM_DIM = 'parm'
    dim_names = CHAN_DIM, FRQ_DIM, PARM_DIM

    def __init__(self, shape, scan_id = None, values = None, ch_names = None, freqs = None, parm_names = None):
        '''Constructor

            Args:
                shape ((int, int, int)): an integer tuple (nchans, nfreqs, nparms)
                scan_id(str or None): the ID of the input record
                values (xarray, ndarray or None): initial values for self.xarray; NaNs if not provided
                ch_names (list of str of None): labels along the `CHAN_DIM` ('chan') dimension
                freqs (list of floats or None): labels along the `FRQ_DIM` ('frq') dimension 
                parm_names (list of str of None): labels along the `PARM_DIM` ('parm') dimension
        '''
        def verify_coords():
            for name, vec, n in zip(
                    ('ch_names', 'freqs', 'parm_names'),
                    (ch_names, freqs, parm_names),
                    shape):
                if vec is not None:
                    if len(vec) != n:
                        raise ValueError(f'Length of {name} should be equal to {n}')

        def set_defaults():
            cn = range(nchans) if ch_names is None else ch_names
            fr = range(nf) if freqs is None else freqs
            pn = range(nparms) if parm_names is None else parm_names

            return cn, fr, pn

        nchans, nf, nparms = shape

        if values is not None:
            if values.shape != shape:
                raise ValueError('values.shape does not match the shape parameter')
        else:
            values = np.empty(shape)
            values[:] = np.nan

        verify_coords()
        ch_names, freqs, parm_names = set_defaults()

        self.scan_id = scan_id
        self.ch_names = ch_names 
        self.freqs = freqs
        self.parm_names = parm_names

        self.data = xr.DataArray(data = values,
                coords = [ch_names, freqs, parm_names],
                dims = self.dim_names,
                name = scan_id)

    def __str__(self):
        s='TFData attributes:\n'
        s+=f'    scan_id: {self.scan_id}\n'
        s+=f'    dim_names: {self.dim_names}\n'
        s+=f'    ch_names: {self.ch_names}\n'
        s+=f'    freqs: {self.freqs}\n'
        s+=f'    parm_names: {self.parm_names}\n'
        s+='Data: '
        s+=self.data.__str__()
        return s

    def write(self, fname):
        ''' Write this TFData object to the .hdf5 file.

        Args:
            fname(str): file name to save to. User is expected to properly
                set the `fname`'s extention to `.hdf5` to reflect the file type.

        Returns:
            Nothing
        '''
        self.data.to_netcdf(path=fname,
                mode='w',
                format='NETCDF4',
                engine='h5netcdf',
                compute=True,
                invalid_netcdf=False)

    @classmethod
    def read(cls, fname):
        ''' Read TFData object from the .hdf5 file created by the write() method.

        Args:
            fname(str): `.hdf5` file name to read from, that was created by a call to
                `TFData.write()` method.

        Returns:
            tfd (TFData): an instance of TFData class reconstructed from the file.
        '''
        xa = xr.load_dataarray(fname, engine = 'h5netcdf')
        coords = xa.coords     # Labels objects along each coordinate

        # Return list of labels along coordinate axis #idim
        labels = lambda idim: list(coords[cls.dim_names[idim]].values)

        tfd = TFData(xa.shape, scan_id = xa.name,
                values = xa,
                ch_names = labels(0),
                freqs = labels(1),
                parm_names = labels(2)
                )

        return tfd

    def get_fixed_f_slice(self, f):
        '''Get a 2D `chans x parms` slice corresponding to some
        frequency value. If the requested frequency does not match
        any of the available frequencies, the closest of available
        frequencies will be chosen.

        Args:
            f (float): requested frequency, Hz

        Returns:
            xa (xarray): `chans x parms` slice
            factual (float): the actual frequency correspondig to the returned
                data, Hz
        '''
        
        kwa = {TFData.FRQ_DIM: f, 'method': 'nearest'}
        xa = self.data.sel(**kwa)
        factual = float(xa.coords[TFData.FRQ_DIM])
        return xa, factual

    def get_nearest_freq(self, f):
        '''Get the nearest actual frequency value for the requested
        frequency.

        Args:
            f (float): requested frequency, Hz

        Returns:
            fnearest (float): the nearest frequency value among
                the frequencies defined for the frequency dimension, Hz
        '''
        return self.get_fixed_f_slice(f)[1]

    def to_pandas(self, chan = None, freq = None):
        '''Select a 2D slice of the TFData data and convert it to a
        pandas dataframe. The slices are either `freqs x parms` 
        for a single channel, or `chans x parms` for a fixed frequency
        value. Specific type is chosen depending on which argument
        (`chan` or `freq`) is given. For fixed frequency slices, the
        frequency closest to the requested among the available ones is
        chosen. 

        Args:
            chan (str or None): the channel name for `freqs x parms`
                slice
            freq (float or None): the frequency value for `chans x parms`
                slice

        Returns:
            df (DataFrame): pandas dataframe
            arg (str or float): the actual value of the argument (`chan` or `freq`)
                used. This is mostly used for frequencies when exact frequency
                values available may not match the requested frequency. For
                fixed channel slices the `chan` argument value is always returned.
        '''
        if (chan is None) and (freq is None):
            raise ValueError('Either chan of freq must be not None')

        if chan is not None:
            return self.data.loc[chan,:,:].to_pandas(), chan

        xa, f = self.get_fixed_f_slice(freq)
        return xa.to_pandas(), f

    # --- end of TFData class definition ----------

# Unit tests
if __name__ == '__main__':
    nchan = 2
    nf = 3
    nparm = 2
    values = np.arange(12).reshape(nchan, nf, nparm)
    ch_names = ['c1', 'c2']
    freqs = [1.1, 2.2, 4.4]
    parm_names = ['mean', 'std']
    fname = 'qq_tfd.hdf5'

    tfd = TFData((nchan, nf, nparm), scan_id = '1234',
            values = values, ch_names = ch_names, freqs = freqs,
            parm_names = parm_names)

    print('tfd:'); print(tfd)

    tfd.write(fname)

    del tfd

    tfd = TFData.read(fname)
    print('\ntfd read from file:'); print(tfd)
   
    """
    print('\ntfd.data[0,2,1]: {}'.format(tfd.data[0,2,1]))
    print('tfd.data[c1,4.4,mean]: {}'.format(tfd.data.loc['c1',4.4,'mean']))
    print('tfd.data[c1,4.4]: {}'.format(tfd.data.loc['c1',4.4]))
    print('tfd.data[chan = c2]: {}'.format(tfd.data.sel(chan = 'c2')))
    print('Axis names:', tfd.dim_names)
    """

    ch = 'c1'
    f = 3.
    print(f'\nChannel = {ch}:')
    print(tfd.to_pandas(chan = ch)[0])

    df, factual = tfd.to_pandas(freq = f)
    print(f'\nf = {factual}:')
    print(df)


