import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from scipy.integrate import cumulative_simpson  # type: ignore
from scipy.special import roots_legendre, roots_genlaguerre  # type: ignore
from scipy.special import roots_hermite, erfc  # type: ignore
from scipy.interpolate import interp1d  # type: ignore

from os.path import splitext
from decimal import Decimal
from rich import print

labelsize = 18
fontsize = 24
font: dict[str, object] = {'family': 'serif',
                           'color': 'black',
                           'weight': 'normal',
                           'size': fontsize,
                           }
bbox = dict(boxstyle='round', edgecolor='#D7D7D7',
            facecolor='white', alpha=0.75)
plt.rc('text', usetex=True)  # type: ignore
plt.rc('font', family='serif')  # type: ignore
plt.rc('text.latex',  # type: ignore
       preamble=r'\usepackage{amsmath,amssymb,amsfonts}')
plt.rc('animation', html='html5')  # type: ignore


def modname(filename: str | None, text: str) -> str | None:
    '''
    Insert text before the file extension of a given filename. If the filename
    has no extension, text is appended to the end.

    Parameters
    ----------
    filename : str
        A valid file name (with or without an extension).
    text : str
        The text to insert before the file extension. If the filename has no
        extension, text is appended to the end.

    Returns
    ----------
    retval : str
        The filename with the text inserted before the file extension or
        appended to the end if no extension exists.
    '''
    if filename is None:
        return None

    idx = filename.index(splitext(filename)[1])
    if idx == 0:
        idx = len(filename)
    return filename[:idx] + text + filename[idx:]


def file_path(pathname: str | None, filename: str | None) -> str | None:
    '''
    Return a fully qualified path to a named file.

    Parameters
    ----------
    pathname : str
        A valid path to a directory / folder, or None.
    filename : str
        A va;lid file name (including an extension where needed),
        or None.

    Returns
    ----------
    retval : str
        If neither pathname nor filename is None, then
        pathname + filename; otherwise None.
    '''
    if (pathname is None) or (filename is None):
        return None
    else:
        return pathname + filename


def figdisp(fig: Figure, show: bool,
            savepath: str | None, filename: str | None):
    '''
    Dispatch a figure: show and/or save

    Parameters
    ----------
    fig : matplotlib.figure.Figure object
        The figure to be displayed/saved.
    show : bool
        If True, show and the close the figure; otherwise,
        close it.
    savepath : str
        A valid path to a directory / folder, or None; if
        None, the figure is not saved.
    filename : str
        A valid file name (including an extension where needed),
        or None; if None, the figure is not saved.
    '''
    filepath = file_path(savepath, filename)
    if filepath is None:
        pass
    else:
        fig.savefig(filepath, bbox_inches='tight')  # type: ignore
        print("Saved {0}\n".format(filepath))

    if show:
        plt.show()  # type: ignore
    else:
        plt.close()  # type: ignore


def classmro(myself: object) -> str:
    ''' A function to parse the __mro__ property of a Python class
        to display its inheritance pedigree. Ambiguous results
        occur when there's multiple inheritance at any level.

    Parameters
    ----------
    myself : class instance
        Usually 'self' if classmro is used within a class

    Returns
    ----------
    retval : str
        A string listing the classes from which 'myself' is derived
        (including 'myself').
    '''
    class_str = myself.__class__.__name__
    for classname in myself.__class__.__mro__[1:-1]:
        class_str += " : {}".format(classname.__name__)
    return class_str


class Storage():
    '''
    A class to store a dictionary of parameters as attributes
    '''
    def __init__(self, **kwargs: object):
        self.__dict__.update(kwargs)


def convert_to_float(value: object) -> float:
    '''
    Convert a value to float if possible, otherwise raise ValueError.

    Parameters
    ----------
    value : object
        The value to convert.

    Returns
    -------
    float
        The converted float value.

    Raises
    ------
    ValueError
        If the value cannot be converted to float.
    '''
    if isinstance(value, (int, float, str)):
        retval = float(value)
    else:
        raise ValueError(
            f"Parameter with value {value} cannot be converted to float."
        )

    return retval


def setarr(x: float | list[float] | npt.NDArray[np.float64],
           count: int | None = None):
    '''
    Convert a number or list of numbers to a NumPy array of float64

    Parameters
    ----------
    x : number, list of numbers, or numpy array
        Convert an input to a numpy array of dtype float64
    count : int
        Length of the returned (1D) array; can be generalized to shape
        in the future; default = None

    Returns
    -------
    retval : numpy.ndarry with dtype('float64') and size = count.
    '''
    if isinstance(x, np.ndarray):
        if count is not None:
            assert x.size == count, \
                'Input array has size {}, not {}.'.format(x.size, count)
        return x.astype(np.float64)
    elif isinstance(x, list):
        if len(x) == 1 and count is not None:
            return np.array(count * x, dtype=np.float64)
        else:
            if count is not None:
                assert len(x) == count, \
                    'Input list has length {}, not {}.'.format(len(x), count)
            return np.array(x, dtype=np.float64)
    else:  # x is a number
        if count is None:
            count = 1
        return np.array(count * [x], dtype=np.float64)


def get_xlim():
    '''
    Return the "nice" values of the min and max of the x-axis of the current
    plot for matplotlib graphics.

    Returns
    ----------
    retval : numpy.ndarray.float64
        The minimum [0] and maximum [1] of the x values, rounded to a value
        that's convenient for a matplotlib graph.
    '''
    loc = plt.xticks()[0]  # type: ignore
    retval = tuple(np.array([loc[0], loc[len(loc)-1]]))  # type: ignore

    return retval


def get_ylim():
    '''
    Return the "nice" values of the min and max of the y-axis of the
    current plot for matplotlib graphics.

    Returns
    ----------
    retval : numpy.ndarray.float64
        The minimum [0] and maximum [1] of the y values, rounded to
        a value that's convenient for a matplotlib graph.
    '''
    loc = plt.yticks()[0]  # type: ignore
    retval = tuple(np.array([loc[0], loc[len(loc)-1]]))  # type: ignore

    return retval


def manexp10(number: float) -> tuple[float, int]:
    '''
    Return the base 10 mantissa and exponent of a floating point number
    for pretty LaTeX printing.

    Parameters
    ----------
    number : numpy.float64

    Returns
    ----------
    a, x : numpy.float64
        The base 10 mantissa and exponent of number.
    '''
    (_, digits, exponent) = Decimal(number).as_tuple()
    assert isinstance(exponent, int), \
        'Exponent of Decimal number is not an integer: {}'.format(exponent)
    x = len(digits) + exponent - 1
    a = float(Decimal(number).scaleb(-x).normalize())

    if (abs(a) >= 10.0):
        x += 1
        a /= 10.0
    elif (abs(a) < 1.0):
        x -= 1
        a *= 10.0

    return a, x


class BaseClass(object):
    '''
    A general-purpose virtual base class

    Required Private Methods
    ------------------------
    _set_specs :
        Dict of keys that will be present in the parameter dictionary
        required by a derived class; default = dict()
    '''

    _specs: dict[str, dict[str, str]]

    def __init__(self, params: dict[str, object]):
        '''
        Specify the dictionary keys of parameters required by the
        derived class, and then set those parameters

        Parameters
        ----------
        params : dict
            A dictionary containing the keys and values of
            parameters required by the derived class
        '''
        self._set_specs()
        self.set_params(params)

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.

        The value of each key will be another dictionary with keys
        'units' : str
            The physical units of the parameter.
        For example, if in the derived class definition:
        def _set_specs(self):
            self._specs = { 'a' : { 'units' : 'mm' },
                            'b' : { 'units' : '' } }
        Then:
            params = {'a' : 1.0, 'b' : 2.0}
            obj = DerivedClass(params)

        The default implementation is an empty dictionary.
        '''
        self._specs = dict()

    def _check_params(self, params: dict[str, object]):
        '''
        Check whether there are any missing or extra parameters
        in params

        Parameters
        ----------
        params : dict
            Dictionary containing a subset of keys specified in
            _set_specs; if any parameters are missing from __dict__,
            they must be included.
        '''
        specs_keys = set(self._specs.keys())
        dict_keys = set(key[1:] for key in self.__dict__.keys())
        params_keys = set(params.keys())

        extra = sorted(params_keys - specs_keys)
        missing = sorted((specs_keys - params_keys)
                         & (specs_keys - dict_keys))

        assert not bool(extra), 'Extra keys in params dict: {}'.format(extra)
        assert not bool(missing), \
            'Missing keys in params dict: {}'.format(missing)

    def __str__(self) -> str:
        '''
        Return a string containing the attributes of an object derived from
        BaseClass. Example:
            class DerivedClass(BaseClass): ...
            obj = DerivedClass(...)
            print(obj)
        '''
        name = self.__dict__.get('_name', None)
        if name is not None:
            retstr: str = (str(name) + ' : ' + classmro(self) + '\n')
        else:
            retstr: str = classmro(self) + '\n'

        for key, _ in self._specs.items():
            if key == 'name':
                continue
            else:
                val = self.__dict__['_' + key]
                if np.abs(val) < 10 * np.finfo(float).eps:
                    retstr += key + ': 0.00'
                elif np.abs(val) >= 0.01:
                    retstr += key + ': {:.2f}'.format(val)
                else:
                    retstr += key + ': {:.2e}'.format(val)
            retstr += ' ' + self._specs[key]['units'] + '\n'

        dict_keys = set(key[1:] for key in self.__dict__.keys())
        specs_keys = set(self._specs.keys())
        specs_keys.add('specs')

        for key in (dict_keys - specs_keys):
            retstr += key + ': ' + self.__dict__['_' + key].__str__()

        return retstr

    def __rich__(self):
        '''
        Return a string containing the attributes of an object derived from
        the BaseClass using rich text. Example:
            from rich import print
            class DerivedClass(BaseClass): ...
            obj = DerivedClass(...)
            print(obj)
        '''
        try:
            name = self.__dict__.get('_name', None)
            if name is not None:
                retstr = "[bold blue]{}[/bold blue]".format(name) \
                    + "[bold cyan] : {}[/bold cyan]\n".format(classmro(self))
            else:
                retstr = "[bold cyan]{}[/bold cyan]\n".format(classmro(self))
        except AttributeError:
            retstr = "[bold cyan]{}[/bold cyan]\n".format(classmro(self))

        retstr += "[green]"
        for key, _ in self._specs.items():
            if key == 'name':
                continue
            else:
                val = self.__dict__['_' + key]
                if np.abs(val) < 10 * np.finfo(float).eps:
                    retstr += key + ': 0.00'
                elif np.abs(val) >= 0.01:
                    retstr += key + ': {:.2f}'.format(val)
                else:
                    retstr += key + ': {:.2e}'.format(val)
            retstr += ' ' + self._specs[key]['units'] + '\n'
        retstr += "[/green]"

        dict_keys = set(key[1:] for key in self.__dict__.keys())
        specs_keys = set(self._specs.keys())
        specs_keys.add('specs')
        retstr += "[red]"
        for key in (dict_keys - specs_keys):
            retstr += key + ': ' + self.__dict__['_' + key].__rich__()
        retstr += "[/red]"

        return retstr

    def set_params(self, params: dict[str, object]):
        '''
        Walk through the list of keyword parameters, and for each one
        create a private variable with a name that is the input parameter name
        preceded by an underscore (i.e., 'varname' becomes '_varname')

        Parameters
        ----------
        params : dict
            A dictionary containing the keys and values of
            parameters required by the derived class
        '''
        self._check_params(params)
        for key, value in params.items():
            self.__dict__['_' + key] = value

    def get_params(self) -> dict[str, object]:
        '''
        Return a dictionary of parameters

        Returns
        -------
        retval : dict
            A dictionary containing the keys and values of
            parameters required by the derived class
        '''
        retval: dict[str, object] = dict()
        for key in self._specs.keys():
            retval[key] = self.__dict__['_' + key]
        return retval


class Shape(BaseClass):
    '''
    A base class for functions of a single variable (e.g., position or
    time). Convenient for ODE solvers such as scipy.integrate.solve_ivp
    and scipy.integrate.solve_bvp.

    Public Methods
    --------------
    shape :
        Function that returns the value of the shape;
        default: raises NotImplementedError
    intshape :
        Function that returns the integral of the shape;
        default: raises NotImplementedError
    plot_shape :
        Plot the shape function
    plot_intshape :
        Plot the integral of the shape function
    '''

    def shape(self, x: float | npt.NDArray[np.float64]) \
            -> float | npt.NDArray[np.float64]:
        '''
        Function that returns the value of the shape; default raises
        NotImplementedError

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable

        Returns
        -------
        numpy.ndarray
            The value of the shape
        '''
        raise NotImplementedError

    def intshape(self, x: npt.NDArray[np.float64]) \
            -> npt.NDArray[np.float64]:
        '''
        Function that returns the integral of the shape; default performs
        a numerical integration using scipy.integrate.cumulative_simpson

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable

        Returns
        -------
        numpy.ndarray
            The cumulative integral of the shape at each value of x
        '''
        #  sx = np.atleast_1d(self.shape(x))
        #  isx = np.zeros_like(sx)
        #  for n in range(1, sx.size):
        #    isx[n] = simpson(sx[0:n], x=x[0:n])
        return cumulative_simpson(
            np.atleast_1d(self.shape(x)), x=x, initial=np.array([0.0])
        ).astype(np.float64)

    def intnorm(self, intval: float, x: npt.NDArray[np.float64]) -> float:
        '''
        Normalize an integral to have a an integrated value of intval at x

        Parameters
        ----------
        intval : numpy.ndarray
            The required value of the integral of the shape at x
        x : float
            The coordinate at which the integral is normalized to intval

        Returns
        -------
        numpy.ndarray
            The normalized integral
        '''
        value = intval / self.intshape(x)[-1]

        return value

    def plot_shape(self, x: npt.NDArray[np.float64], show: bool = True,
                   savepath: str | None = None, filename: str | None = None):
        '''
        Plot the shape function

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable
        show : bool
            If True, display the plot; otherwise, do not display
        savepath : str
            A valid path to a directory / folder, or None; if
            None, the figure is not saved.
        filename : str
            A valid file name (including an extension), or None;
            if None, the figure is not saved.
        '''
        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        ax.plot(x, self.shape(x))  # type: ignore
        ax.set_xlabel(r'$x$', fontdict=font)  # type: ignore
        ax.set_ylabel(r'$S(x)$', fontdict=font)  # type: ignore
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(*get_ylim())

        figdisp(fig, show, savepath, filename)

    def plot_intshape(self, x: npt.NDArray[np.float64], show: bool = True,
                      savepath: str | None = None,
                      filename: str | None = None):
        '''
        Plot the integral of the shape function

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable
        show : bool
            If True, display the plot; otherwise, do not display
        savepath : str
            A valid path to a directory / folder, or None; if
            None, the figure is not saved.
        filename : str
            A valid file name (including an extension), or None;
            if None, the figure is not saved.
        '''
        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        ax.plot(x, self.intshape(x))  # type: ignore
        ax.set_xlabel(r'$x$', fontdict=font)  # type: ignore
        ax.set_ylabel(r'$\int_0^x d x^\prime\, S(x^\prime)$',  # type: ignore
                      fontdict=font)
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(*get_ylim())

        figdisp(fig, show, savepath, filename)


class Const(Shape):
    '''
    A constant shape function.

    Usage
    -----
    Define a constant shape function with value = 1.0:
    params = {'name' : 'Const Shape Example', 'value' : 1.0}
    obj = Const(params)
    '''

    _value: float
    _norm: float

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        '''
        self._specs = {'name': {'units': ''},   # name of the object
                       'value': {'units': ''},  # constant value
                       'norm': {'units': ''}    # normalization factor
                       }

    def shape(self, x: float | npt.NDArray[np.float64]) \
            -> float | npt.NDArray[np.float64]:
        '''
        Function that returns the shape.

        Parameters
        ----------
        x : float or numpy.ndarray
            The independent variable

        Returns
        -------
        float or numpy.ndarray
            The value of the shape
        '''
        return self._value * self._norm * np.ones_like(x)

    def intshape(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        '''
        Function that returns the integral of the shape.

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable

        Returns
        -------
        numpy.ndarray
            The integral of the shape
        '''
        return self._value * self._norm * x


class LinearDrop(Shape):
    '''
    A linear drop shape function that begins at x = 0 with a constant
    value and drops linearly to 0.

    Usage
    -----
    Define a linear drop shape function with value = 1.0 and delta = 1.0:
    params = {'name' : 'Linear Drop Shape Example', 'value' : 1.0,
              'delta' : 1.0}
    obj = LinearDrop(params)
    '''

    _value: float
    _norm: float
    _delta: float

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        '''
        self._specs = {'name': {'units': ''},   # name of the object
                       'value': {'units': ''},   # constant value at t = 0
                       'norm': {'units': ''},    # normalization factor
                       'delta': {'units': ''}    # linear drop rate to 0
                       }

    def shape(self, x: float | npt.NDArray[np.float64]) \
            -> float | npt.NDArray[np.float64]:
        '''
        Function that returns the shape.

        Parameters
        ----------
        x : float or numpy.ndarray
            The independent variable

        Returns
        -------
        float or numpy.ndarray
            The value of the shape
        '''
        return self._value * self._norm * np.maximum(0.0, 1 - x / self._delta)

    def intshape(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        '''
        Function that returns the integral of the shape.

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable

        Returns
        -------
        numpy.ndarray
            The integral of the shape
        '''
        return (self._value * self._norm
                * np.minimum(self._delta / 2, x - 0.5 * x**2 / self._delta))


class TrapRight(Shape):
    '''
    A right-trapezoidal shape function that begins at x = 0 with a constant
    value until x = x_2, and then drops linearly to 0.

    Usage
    -----
    Define a right-trapezoidal shape function with value = 1.0, x_2 = 1.0,
    and delta_2 = 1.0:
    params = {'name' : 'Trap Right Shape Example', 'value' : 1.0, 'x_2' : 1.0,
    'delta_2' : 1.0}
    obj = TrapRight(params)
    '''

    _value: float
    _norm: float
    _x_2: float
    _delta_2: float

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        '''
        self._specs = {'name': {'units': ''},   # name of the object
                       'value': {'units': ''},   # constant value until x_2
                       'norm': {'units': ''},    # normalization factor
                       'x_2': {'units': ''},     # time until drop
                       'delta_2': {'units': ''}  # linear drop rate to 0
                       }

    def shape(
        self, x: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Function that returns the shape. Note that we assume that delta_2 is
        large enough to allow solvers to converge but small enough that the
        shape is approximately a rectangle.

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable

        Returns
        -------
        numpy.ndarray
            The value of the shape
        '''
        x_2 = self._x_2
        d_2 = self._delta_2

        sx = self._value * self._norm * (
            np.heaviside(x, 1.0) * np.heaviside(x_2 - d_2/2 - x, 1.0)
            + (1 - (x - (x_2 - d_2/2))/d_2)
            * np.heaviside(x - (x_2 - d_2/2), 0.0)
            * np.heaviside(x_2 + d_2/2 - x, 0.0)
        )

        return sx

    def intshape(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        '''
        Function that returns the integral of the shape. Note that we assume
        that delta_2 is large enough to allow solvers to converge but small
        enough that the shape is approximately a rectangle.

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable

        Returns
        -------
        numpy.ndarray
            The integral of the shape
        '''
        x_2 = self._x_2

        isx = (
            self._value
            * self._norm
            * (
                np.heaviside(x, 1.0) * np.heaviside(x_2 - x, 1.0)
                + np.heaviside(x - x_2, 0.0)
            )
        )

        return isx


class TrapAcute(Shape):
    '''
    A trapezoidal shape function that rises linearly from 0 to a constant value
    at x = x_1, remains constant until x = x_2, and then drops linearly to 0.

    Usage
    -----
    Define a trapezoidal shape function with value = 1.0, x_1 = 1.0, x_2 = 2.0,
    delta_1 = 1.0, and delta_2 = 1.0:
    params = {
        'name': 'Trap Acute Shape Example',
        'value': 1.0,
        'x_1': 1.0,
        'x_2': 2.0,
        'delta_1': 1.0,
        'delta_2': 1.0
    }
    obj = TrapAcute(params)
    '''
    _value: float
    _norm: float
    _x_1: float
    _x_2: float
    _delta_1: float
    _delta_2: float

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        '''
        self._specs = {
            'name': {'units': ''},     # name of the object
            'value': {'units': ''},    # constant value for x_1 < x < x_2
            'norm': {'units': ''},     # normalization factor
            'x_1': {'units': ''},      # time to rise from 0 to value
            'x_2': {'units': ''},      # time to drop from value to 0
            'delta_1': {'units': ''},  # linear rise rate from 0
            'delta_2': {'units': ''}   # linear drop rate to 0
        }

    def shape(
        self, x: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Function that returns the shape. Note that we assume that delta_1 and
        delta_2 are large enough to allow solvers to converge but small enough
        that the shape is approximately a rectangle.

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable

        Returns
        -------
        numpy.ndarray
            The value of the shape
        '''
        x_1 = self._x_1
        x_2 = self._x_2
        d_1 = self._delta_1
        d_2 = self._delta_2

        sx = self._value * self._norm * (
            ((x - (x_1 - d_1/2)) / d_1)
            * np.heaviside(x - (x_1 - d_1/2), 0.0)
            * np.heaviside(x_1 + d_1/2 - x, 0.0)
            + np.heaviside(x - (x_1 + d_1/2), 1.0)
            * np.heaviside(x_2 - d_2/2 - x, 1.0)
            + (1 - (x - (x_2 - d_2/2))/d_2)
            * np.heaviside(x - (x_2 - d_2/2), 0.0)
            * np.heaviside(x_2 + d_2/2 - x, 0.0)
        )

        return sx

    def intshape(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        '''
        Function that returns the integral of the shape. Note that we assume
        that delta_1 and delta_2 are large enough to allow solvers to converge
        but small enough that the shape is approximately a rectangle.

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable

        Returns
        -------
        numpy.ndarray
            The integral of the shape
        '''
        x_1 = self._x_1
        x_2 = self._x_2

        isx = (
            self._value * self._norm
            * (
                (x - x_1)
                * np.heaviside(x - x_1, 1.0)
                * np.heaviside(x_2 - x, 1.0)
                + (x_2 - x_1)
                * np.heaviside(x - x_2, 0.0)
            )
        )

        return isx


class Gaussian(Shape):
    '''
    A Gaussian shape function with value, width, and x_max:
    S(x) = value * * exp(-2 * ((x - x_max) / width)**2)

    Usage
    -----
    Define a Gaussian shape function with value = 1.0, width = 1.0, and
    x_max = 0.0:
    params = {'name' : 'Gaussian Shape Example', 'value' : 1.0, 'norm' : 1,
              'width' : 1.0, 'x_max' : 0.0}
    obj = Gaussian(params)
    '''

    _value: float
    _norm: float
    _width: float
    _x_max: float

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        '''
        self._specs = {'name': {'units': ''},   # name of the object
                       'value': {'units': ''},  # gaussian pulse scale factor
                       'norm': {'units': ''},   # normalization factor
                       'width': {'units': ''},  # 1/e**2 pulse width
                       'x_max': {'units': ''}   # pulse max
                       }

    def shape(
        self, x: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Function that returns the shape; note that the Gaussian is not
        normalized to have unit area: set the norm parameter to
        (np.sqrt(2/np.pi)/self._width) to normalize the Gaussian to unit area.

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable

        Returns
        -------
        numpy.ndarray
            The value of the shape
        '''
        return (
            self._value
            * self._norm
            * np.exp(-2 * ((x - self._x_max) / self._width) ** 2)
        )

    def intshape(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        '''
        Function that returns the integral of the shape

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable

        Returns
        -------
        numpy.ndarray
            The integral of the shape
        '''
        norm = self._norm * (self._width / np.sqrt(2 / np.pi))
        return np.array(
            (self._value / 2)
            * norm
            * erfc(np.sqrt(2) * (x - self._x_max) / self._width),
            dtype=np.float64
        )


class Sine(Shape):
    '''
    A sine shape function with offset, amplitude, frequency, and phase:
    S(x) = offset + amplitude * sin(2 * pi * frequency * x + phase)

    Usage
    -----
    Define a sine shape function with offset = 0.0, amplitude = 1.0,
    frequency = 1.0, and phase = 0.0:
    params = {'name' : 'Sine Shape Example', 'offset' : 0.0,
              'amplitude' : 1.0, 'frequency' : 1.0, 'phase' : 0.0}
    obj = Sine(params)
    '''

    _offset: float
    _amplitude: float
    _frequency: float
    _phase: float

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        '''
        self._specs = {
            'name': {'units': ''},       # name of the object
            'offset': {'units': ''},     # function oscillates about this value
            'amplitude': {'units': ''},  # amplitude of the sine function
            'frequency': {'units': ''},  # oscillation frequency (units of 1/t)
            'phase': {'units': ''}       # phase of the sine function at t = 0
        }

    def shape(
        self, x: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        return (
            self._offset
            + self._amplitude
            * np.sin(2 * np.pi * self._frequency * x + self._phase)
        )

    def intshape(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return (
            self._offset * x
            - (self._amplitude / (2 * np.pi * self._frequency))
            * np.cos(2 * np.pi * self._frequency * x + self._phase)
        )


def findkink(
    index_list: npt.NDArray[np.int32],
    phi: npt.NDArray[np.float64],
    q: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    '''
    Try to find a kink in the first derivative of a phase angle
    that has been unwrapped by numpy; currently, the search stops
    after one kink

    Parameters
    ----------
    index_list : numpy.ndarray.int32
        An array of integers containing indices of phi.
    phi : numpy.ndarray.float64
        An array of numpy-unwrapped angles.
    q : numpy.ndarray.int32
        An array of integers that will be multiplied by 2 * pi to produce
        samples for extrapolation comparison. As a rule, the largest entry
        in q should be greater than the largest entry in index_list.

    Returns
    ----------
    kink_list : numpy.ndarray.int32
        A list of indices where kinks were found.
    '''
    kink_list: list[int] = []
    phiu = np.zeros_like(phi)
    phiu[index_list[0]:index_list[2]] = phi[index_list[0]:index_list[2]]
    for ndx in index_list[2:]:
        xp = index_list[:ndx]
        fp = phiu[:ndx]
        f = interp1d(xp, fp, bounds_error=False,
                     fill_value='extrapolate')  # type: ignore
        pred = f(float(ndx))  # type: ignore
        comp = phi[ndx] + 2 * q * np.pi
        idx_min = int(np.argmin(np.abs(comp - pred)))  # type: ignore
        phiu[int(ndx)] = comp[idx_min]
        if np.argmin(np.abs(comp - pred)) != np.max(q):  # type: ignore
            kink_list.append(ndx - 1)
            break
    return np.array(kink_list)


def unwrapcd(
    index_list: npt.NDArray[np.int32],
    phi: npt.NDArray[np.float64],
    q: npt.NDArray[np.int32]
) -> npt.NDArray[np.float64]:
    '''
    Try to repair kinks in the first derivative of a phase angle
    that has been unwrapped by numpy

    Parameters
    ----------
    index_list : numpy.ndarray.int32
        An array of integers containing indices of phi.
    phi : numpy.ndarray.float64
        An array of numpy-unwrapped angles.
    q : numpy.ndarray.int32
        An array of integers that will be multiplied by 2 * pi to produce
        samples for extrapolation comparison. As a rule, the largest entry
        in q should be greater than the largest entry in index_list.

    Returns
    ----------
    phiu : numpy.ndarray.float64
        An array containing (kink-free) unwrapped angles.
    '''
    phiu = np.zeros_like(phi)
    phiu[index_list[0]:index_list[2]] = phi[index_list[0]:index_list[2]]
    for ndx in index_list[2:]:
        xp = index_list[:ndx]
        fp = phiu[:ndx]
        f = interp1d(xp, fp, bounds_error=False,
                     fill_value='extrapolate')  # type: ignore
        pred = f(float(ndx))  # type: ignore
        comp = phi[ndx] + 2 * q * np.pi
        phiu[ndx] = comp[np.argmin(np.abs(comp - pred))]  # type: ignore
    return phiu


# def is_convertible_to_float(value: object | None) -> bool:
#     if value is None:
#         return False
#     try:
#         if isinstance(value, (int, float, str)):
#             return True
#         else:
#             return False
#     except (ValueError, TypeError):
#         return False


def gauss_legendre(
    n: int, **kwargs: object
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    '''
    Compute the abscissae (x) and weights (w) for Gauss-Legendre quadrature. An
    example of typical usage is:
        x, w = gauss_legendre(npts, a = 0.0, b = 0.5)

    Parameters
    ----------
    n : numpy.int32
        An integer > 0 giving the number of abscissae and weights; if n is odd,
        the midpoint of the interval will be included.
    a, b : numpy.float64
        Real numbers giving the integration interval [a, b]; if these arguments
        are absent, then the default interval [-1, +1] is used.

    Returns
    ----------
    x : numpy.ndarray.float64
        A vector containing the abscissae.
    w : numpy.ndarray.float64
        A vector containing the weights.
    '''
    x_0, w_0 = roots_legendre(n)  # type: ignore

    if kwargs:
        a = convert_to_float(kwargs.get('a', -1.0))
        b = convert_to_float(kwargs.get('b', 1.0))

        x = np.array(0.5 * ((b - a) * x_0 + (a + b)), dtype=np.float64)
        w = np.array(0.5 * (b - a) * w_0, dtype=np.float64)
    else:
        x = np.array(x_0, dtype=np.float64)
        w = np.array(w_0, dtype=np.float64)

    return (x, w)


def gauss_hermite(
    n: int, **kwargs: object
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    '''
    Compute the abscissae (x) and weights (w) for Gauss-Hermite quadrature. An
    example of typical usage is:
        x, w = gauss_hermite(npts, m = 0.5, sigma = 2.0)

    Parameters
    ----------
    n : numpy.int32
        An integer > 0 giving the number of abscissae and weights; if n is odd,
        the midpoint of the interval will be included.
    m, sigma : numpy.float64
        Real numbers defining the generalized weight function
        exp(-(x - m)**2/sigma**2); if these arguments are absent, then the
        default values m = 0 and sigma = 1 are used.

    Returns
    -------
    x : numpy.ndarray.float64
        A vector containing the abscissae.
    w : numpy.ndarray.float64
        A vector containing the weights.
    '''
    x_0, w_0 = roots_hermite(n)  # type: ignore

    if kwargs:
        sigma = convert_to_float(kwargs.get('sigma', 1.0))
        m = convert_to_float(kwargs.get('m', 0.0))
        x = np.array(sigma * x_0 + m, dtype=np.float64)
        w = np.array(sigma * w_0, dtype=np.float64)
    else:
        x = np.array(x_0, dtype=np.float64)
        w = np.array(w_0, dtype=np.float64)

    return (x, w)


def gauss_laguerre(
    n: int, **kwargs: object
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    '''
    Compute the abscissae (x) and weights (w) for generalized Gauss-Laguerre
    quadrature. An example of typical usage is:
        x, w = gauss_laguerre(npts, a = 1.0, b = 0.5)

    Parameters
    ----------
    n : numpy.int32
        An integer > 0 giving the number of abscissae and weights; if n is odd,
        the midpoint of the interval will be included.
    a, b : numpy.float64
        Real numbers defining the generalized weight function x**a*exp(-b*x);
        if a > -1 is absent, then the default value a = 0 is used; if b is
        absent, then the default value b = 1 is used.

    Returns
    ----------
    x : numpy.ndarray.float64
        A vector containing the abscissae.
    w : numpy.ndarray.float64
        A vector containing the weights.
    '''
    a = convert_to_float(kwargs.get('a', 0.0))
    assert a > -1, \
        "Parameter 'a' must be greater than -1 for Gauss-Laguerre quadrature."
    x_0, w_0 = roots_genlaguerre(n, a)  # type: ignore

    if kwargs:
        b = convert_to_float(kwargs.get('b', 1.0))
        x = x_0 / b
        w = w_0 / b**(a + 1)
    else:
        x = np.array(x_0, dtype=np.float64)
        w = np.array(w_0, dtype=np.float64)

    return (x, w)
