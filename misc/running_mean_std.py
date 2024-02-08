"""
Calculate mean and STD of a stream of data.

Source:  
    https://www.johndcook.com/blog/standard_deviation/
    https://stackoverflow.com/questions/895929/how-do-i-determine-the-standard-deviation-stddev-of-a-set-of-values
"""
import numpy as np

class RunningMeanSTD:
    """
    This class' purpose is to calculate a running value of the mean and
    variance (or standard deviation) of a stream of data. Each data item
    may be a scalar or a numpy ndarray.

    Methods:
    """
    _np_types = [np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc,
            np.uintc,  np.int_, np.uint, np.longlong, np.ulonglong, np.half,
            np.float16, np.single, np.double, np.longdouble,
            np.csingle, np.cdouble, np.clongdouble]

    def __init__(self):
        """Constructor
        """
        self._n = 0         # Number of currently accumulated values
        self._oldM = None   # Old value for mean
        self._newM = None   # New value for mean
        self._oldS = None   # Old value for variance
        self._newS = None   # New value for variance

        # These are needed to avoid excessive memory reallocations
        self._bufM = self._newM # Buffer pointer for mean
        self._bufS = self._newS # Buffer pointer for variance

        self._array = False     # Flag that value is a matrix

    def reset(self):
        """
        Clear all accumulated data and reinitialize the object
        """
        self._n = 0         # Number of currently accumulated values

        if self._oldM is not None:
            del self._oldM; self._oldM = None

        if self._newM is not None:
            del self._newM; self._newM = None

        if self._oldS is not None:
            del self._oldS; self._oldS = None

        if self._newS is not None:
            del self._newS; self._newS = None

        self._bufM = self._newM
        self._bufS = self._newS
        self._array = False

    @classmethod
    def is_array(cls, v):
        """
        Args:
            v: data item from the stream

        Returns
            `True` for non-scalar data types, `False` otherwise
        """

        if type(v) in (int, float, bool, *cls._np_types):
            return False

        return True

    @classmethod
    def _check_input_type(cls, v):
        """Verify that input type is supported.

        Args:
            v: data item from the stream

        Returns
            `True` for valid data types, `False` otherwise
        """
        if type(v) in (int, float, bool, np.ndarray, *cls._np_types):
            return True

        return False

    def push(self, v):
        """Add new data item `v` from the stream to accumulated statistics.
        All the magic happens here.

        Args:
            v: data item from the stream

        Returns:
            nothing
        """
        self._n += 1

        if self._n == 1:
            if not self._check_input_type(v):
                raise ValueError('Input data type is not supported')

            # Set stats for a single data value, enforcing conversion to float
            # of proper shape for a matrix variable
            if self.is_array(v):
                self._array = True
                self._oldM = v.copy().astype(float)
                self._newM = v.copy().astype(float)

                # Allocate memory chunks of correct size and set all elements to 0
                self._oldS = np.zeros(v.shape, dtype = float)
                self._newS = np.zeros(v.shape, dtype = float)

                # Ptrs to where the updated results will go after push()
                self._bufM = self._newM
                self._bufS = self._newS
            else:
                self._array = False
                self._oldM = v
                self._newM = v
                self._oldS = 0.
                self._newS = 0.
        else:
            if self._array:
                np.add(self._oldM, (v - self._oldM)/self._n, out = self._bufM)
                self._newM = self._bufM     # These carry current running result

                np.add(self._oldS, (v - self._oldM)*(v - self._newM), out = self._bufS)
                self._newS = self._bufS

                self._bufM = self._oldM     # These will be overwritten on the next push()
                self._bufS = self._oldS

                self._oldM = self._newM
                self._oldS = self._newS
            else:
                self._newM = self._oldM + (v - self._oldM)/self._n
                self._newS = self._oldS + (v - self._oldM)*(v - self._newM)
                self._oldM = self._newM
                self._oldS = self._newS

    def counter(self):
        """Return current number of accumulated data points
        """
        return self._n

    def mean(self):
        """Return currently accumulated average
        """
        return self._newM

    def var(self):
        """Return currently accumulated variance
        """
        return self._newS/(self._n - 1) if self._n > 1 else 0

    def std(self):
        """Return currently accumulated standard deviation
        """
        return np.sqrt(self.var())

# Unit tests
if __name__ == '__main__':
    summator = RunningMeanSTD()

    seed = 12345
    rng = np.random.default_rng(seed)

    # Verify on scalars: floats
    n = 10000
    x = rng.random(n)

    for xi in x:
        summator.push(xi)

    assert np.allclose(np.mean(x), summator.mean())
    assert np.allclose(np.std(x, ddof = 1), summator.std())
    assert np.allclose(np.var(x, ddof = 1), summator.var())

    # Verify on scalars: integers
    n = 10000
    x = 1000*rng.random(n)
    x = x.astype(int)
    summator.reset()

    for xi in x:
        summator.push(xi)

    assert np.allclose(np.mean(x), summator.mean())
    assert np.allclose(np.std(x, ddof = 1), summator.std())
    assert np.allclose(np.var(x, ddof = 1), summator.var())

    # Verify on matrices: floats
    x = rng.random((100,3,4,5))
    summator.reset()

    for xi in x:
        summator.push(xi)

    assert np.allclose(np.mean(x, axis = 0), summator.mean())
    assert np.allclose(np.std(x, axis = 0, ddof = 1), summator.std())
    assert np.allclose(np.var(x, axis = 0, ddof = 1), summator.var())

    # Check integer arrays
    x = np.arange(256).reshape(16,16).astype(int)
    summator.reset()

    for xi in x:
        summator.push(xi)

    assert np.allclose(np.mean(x, axis = 0), summator.mean())
    assert np.allclose(np.std(x, axis = 0, ddof = 1), summator.std())
    assert np.allclose(np.var(x, axis = 0, ddof = 1), summator.var())

    print('All tests passed')

