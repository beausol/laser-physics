import numpy as np
import numpy.typing as npt

from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import brentq, fmin
from scipy.special import erf

import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from matplotlib import animation

from laser.odeintegrator import ODEIntegrator

from laser.utils import (
    BaseClass, Shape, classmro, Const, TrapAcute, Storage, setarr,
    font, fontsize, labelsize, bbox,
    get_ylim, modname, figdisp, file_path,
    convert_to_float
)

from typing import Any
from enum import Enum, auto
from copy import deepcopy
from rich import print


class Config(Enum):
    '''
    Enumeration of laser configurations.

    Attributes
    ----------
    ULA : auto()
        Unidirectional laser amplifier
    URL : auto()
        Unidirectional ring laser
    SWL : auto()
        Standing-wave laser

    Methods
    -------
    kappa()
        Return the gain saturation factor kappa
    zlims()
        Return a tuple specifying the minimum and maximum values of the
        normalized longitudinal coordinate z: (0.0, 1/kappa)
    '''
    ULA = auto()    # unidirectional laser amplifier
    URL = auto()    # unidirectional ring laser
    SWL = auto()    # standing-wave laser

    def __str__(self):
        return classmro(self) + '\n' + 'name: ' + self.name + '\n'

    def __rich__(self):
        return (
            "[green]" + classmro(self) + '\n'
            + 'name: ' + "[/green]" + "[bold green]" + self.name
            + "[/bold green]" + '\n'
        )

    def kappa(self):
        '''
        Return the gain saturation factor kappa.

        Returns
        -------
        numpy.float64
            Gain saturation factor kappa
        '''
        if self.name == 'ULA' or self.name == 'URL':
            return 1.0
        else:
            return 2.0

    def zlims(self):
        '''
        Return a tuple specifying the minimum and maximum values of the
        normalized longitudinal coordinate z: (0.0, 1/kappa).

        Returns
        -------
        tuple
            Tuple containing the minimum and maximum values of the normalized
            longitudinal coordinate z: (0.0, 1/kappa)
        '''
        return (0.0, 1/self.kappa())


class LineShape(BaseClass):
    '''
    Virtual base class for laser gain lineshape objects.

    Parameters
    ----------
    params : dict
        Dictionary containing the following key-value pairs needed
        to initialize a LineShape object; the minimum required keys are:
            'name' : str
                Name of the object
            'omega' : numpy.float64
                Normalized detuning Omega

    Methods
    -------
    rho()
        Return the real part of the lineshape at the normalized detuning Omega
    iota()
        Return the imaginary part of the lineshape at the normalized
        detuning Omega
    annotation()
        Return the annotation for the lineshape object
    '''

    _omega: float | npt.NDArray[np.float64]  # Normalized detuning Omega

    def __init__(self, params: dict[str, object]):
        '''
        Initialize a LineShape object.

        Parameters
        ----------
        params : dict
            Dictionary containing the following key-value pairs needed
            to initialize a LineShape object; the minimum required keys are:
                'name' : str
                    Name of the object
                'omega' : numpy.float64
                    Normalized detuning Omega

        Raises
        ------
        AssertionError
            If the real part of the lineshape at the normalized detuning
            Omega = 0 is not 1.0.
        '''
        BaseClass.__init__(self, params)
        self._check_rho()

    def annotation(self) -> str:
        '''
        Return the annotation for the lineshape object.

        Returns
        -------
        str
            String containing the annotation
        '''
        raise NotImplementedError("Subclasses must implement annotation()")

    def _lineshape(
        self,
        omega: float | npt.NDArray[np.float64]
    ) -> np.complex128:
        '''
        Return the complex lineshape at the normalized detuning Omega.

        Parameters
        ----------
        omega : numpy.float64 or numpy.ndarray
            Normalized detuning Omega

        Returns
        -------
        numpy.complex128
            Complex lineshape at the normalized detuning Omega

        Raises
        ------
        NotImplementedError
        '''
        raise NotImplementedError

    def _check_rho(self):
        '''
        Check that the real part of the lineshape at the normalized detuning
        Omega = 0 is 1.0.

        Raises
        ------
        AssertionError
            If the real part of the lineshape at the normalized detuning
            Omega = 0 is not 1.0.
        '''
        val = self._lineshape(0.0)
        assert (
            np.abs(np.real(val) - 1.0) < 10 * np.finfo(float).eps
        ), (
            "Re[L(0)] = {}; must be 1.0.".format(np.real(val))
        )

    def rho(self):
        '''
        Return the real part of the lineshape function at the normalized
        detuning Omega.

        Returns
        -------
        numpy.float64
            Real part of the lineshape function at the normalized
            detuning Omega
        '''
        return np.real(self._lineshape(self._omega))

    def iota(self):
        r'''
        Return the imaginary part of the lineshape function at the normalized
        detuning Omega.

        Returns
        -------
        numpy.float64
            Imaginary part of the lineshape function at the normalized
            detuning Omega
        '''
        return np.imag(self._lineshape(self._omega))


class Lorentzian(LineShape):
    '''
    Class for a Lorentzian gain lineshape.

    Parameters
    ----------
    params : dict
        Dictionary containing the following key-value pairs needed
        to initialize a Lorentzian object:
            'name' : str
                Name of the object
            'omega' : numpy.float64
                Normalized detuning Omega

    Methods
    -------
    annotation()
        Return the annotation for the Lorentzian object
    '''
    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        '''
        self._specs = {'name': {'units': ''},   # name of the object
                       'omega': {'units': ''}    # normalized detuning Omega
                       }

    def _lineshape(
        self,
        omega: float | npt.NDArray[np.float64]
    ) -> np.complex128:
        '''
        Return the complex Lorentzian lineshape at the normalized
        detuning Omega.

        Parameters
        ----------
        omega : numpy.float64
            Normalized detuning Omega

        Returns
        -------
        numpy.complex128
            Complex Lorentzian lineshape at the normalized detuning Omega
        '''
        return np.complex128(1 / (1 - 1j * omega))

    def annotation(self):
        '''
        Return the annotation for the Lorentzian object.

        Returns
        -------
        tuple
            Tuple containing the x-coordinate, y-coordinate, and the annotation
        '''
        return r'$\Omega = {}$'.format(self._omega)


class AsymmetricLEF(LineShape):
    '''
    Class for an asymmetric gain lineshape based on a Lorentzian + a linewidth
    enhancement factor.

    Parameters
    ----------
    params : dict
        Dictionary containing the following key-value pairs needed
        to initialize an AsymmetricLEF object:
            'name' : str
                Name of the object
            'omega' : numpy.float64
                Normalized detuning Omega
            'lef' : numpy.float64
                Linewidth enhancement factor alpha_LEF

    Methods
    -------
    annotation()
        Return the annotation for the AsymmetricLEF object
    '''

    _lef: (
        float | npt.NDArray[np.float64]
    )  # Linewidth enhancement factor alpha_LEF

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        '''
        self._specs = {
            'name': {'units': ''},    # name of the object
            'omega': {'units': ''},   # normalized detuning Omega
            'lef': {'units': ''}      # linewidth enhancement factor \alpha_LEF
        }

    def _lineshape(
        self,
        omega: float | npt.NDArray[np.float64]
    ) -> np.complex128:
        '''
        Return the complex asymmetric lineshape at the normalized
        detuning Omega.

        Parameters
        ----------
        omega : numpy.float64
            Normalized detuning Omega

        Returns
        -------
        numpy.complex128
            Complex asymmetric lineshape at the normalized detuning Omega
        '''
        return np.complex128(
            (1 - 1j * self._lef) ** 2
            / (1 - 1j * (omega + self._lef))
        )

    def annotation(self) -> str:
        '''
        Return the annotation for the AsymmetricLEF object.

        Returns
        -------
        str
            String containing the annotation
        '''
        return (r'$\Omega = {}$'.format(self._omega) + '\n'
                + r'$\alpha_\mathrm{{LEF}} = {}$'.format(self._lef))


class Gain(BaseClass):
    '''
    Virtual base class for laser gain objects.

    Parameters
    ----------
    params : dict
        Dictionary containing the key-value pairs needed
        to initialize a Gain object; the minimum required keys are:
            'name' : str
                Name of the object
            'gbar_0' : numpy.float64
                Integrated unsaturated gain in the amplifier or laser
        Additional key-value pairs can be included in the dictionary
        to set the parameters of the Gain object
    config : Config
        Configuration of the laser system
    gainz : Shape
        Shape object representing the gain profile
    lineshape : LineShape
        LineShape object representing the lineshape

    Methods
    -------
    set_params(params)
        Set the parameters of the Gain object
    check_config(config)
        Check that the configuration of another Gain object is compatible
        with the configuration of the laser system
    gain(z)
        Return the gain value at the longitudinal coordinate z
    intgain(z)
        Return the integrated gain up to the longitudinal coordinate z
    rho()
        Return the real part of the lineshape at the normalized detuning Omega
    iota()
        Return the imaginary part of the lineshape at the normalized
        detuning Omega
    ls_annotation()
        Return the annotation for the lineshape object
    plot_gain(z, show=True, savepath=None, filename=None)
        Plot the gain over the longitudinal coordinates z
    plot_intgain(z, show=True, savepath=None, filename=None)
        Plot the integrated gain over the longitudinal coordinates z
    '''

    _gbar_0: float  # Integrated unsaturated gain
    _z_min: float   # Minimum value of the normalized longitudinal coordinate z
    _z_max: float   # Maximum value of the normalized longitudinal coordinate z

    def __init__(
        self,
        params: dict[str, object],
        config: Config,
        gainz: Shape,
        lineshape: LineShape
    ):
        '''
        Initialize a Gain object.

        Parameters
        ----------
        params : dict
            Dictionary containing the following key-value pairs needed
            to initialize a Gain object; the minimum required keys are:
                'name' : str
                    Name of the object
                'gbar_0' : numpy.float64
                    Integrated unsaturated gain in the amplifier or laser
            Additional key-value pairs can be included in the dictionary
            to set the parameters of the Gain object
        config : Config
            Configuration of the laser system
        gainz : Shape
            Shape object representing the gain profile
        lineshape : LineShape
            LineShape object representing the lineshape
        '''
        self._set_specs()
        self._specs['z_min'] = {'units': ''}
        self._specs['z_max'] = {'units': ''}

        params['z_min'] = config.zlims()[0]
        params['z_max'] = config.zlims()[1]

        self._config = deepcopy(config)
        self._gainz = deepcopy(gainz)
        self._lineshape = deepcopy(lineshape)
        self.set_params(params)

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        '''
        self._specs = {
            'name': {'units': ''},    # name of the object
            'gbar_0': {'units': ''}   # integrated unsaturated gain in the
                                      # amplifier or laser
        }

    def set_params(self, params: dict[str, object]):
        '''
        Set the parameters of the Gain object.

        Parameters
        ----------
        params : dict
            Dictionary containing the parameters to set; the keys must be
            present in the _specs dictionaries of the Gain, LineShape, and/or
            Shape objects. If the name key is present in the params dictionary,
            and it matches the name of either the Shape or LineShape object,
            the parameters are set in the corresponding object. Otherwise, the
            parameters are set in the attributes of the Gain object

        Note
        ----
        The names of the Shape and LineShape objects can't be changed.
        '''
        if (
            'name' in params
            and params['name'] == self._gainz.get_params()['name']
        ):
            del params['name']
            self._gainz.set_params(params)
        elif (
            'name' in params
            and params['name'] == self._lineshape.get_params()['name']
        ):
            del params['name']
            self._lineshape.set_params(params)
        else:
            BaseClass.set_params(self, params)
            if 'gbar_0' in params and isinstance(params['gbar_0'], float):
                self._gainz.set_params({'value': self._gbar_0})

    def check_config(self, config: Config):
        '''
        Check that the configuration of another Gain object is compatible with
        the configuration of the laser system.

        Parameters
        ----------
        config : Config
            Configuration of the other Gain object

        Raises
        ------
        AssertionError
            If the configurations are incompatible
        '''
        assert self._config == config, (
            "Incompatible configurations: {} and {}."
            .format(self._config.name, config.name)
        )

    def gain(self, z: npt.NDArray[np.float64]):
        '''
        Return the gain value at the longitudinal coordinate z.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinate z

        Returns
        -------
        numpy.ndarray
            Gain value at the longitudinal coordinate z
        '''
        return self._gainz.shape(z)

    def intgain(self, z: npt.NDArray[np.float64]):
        '''
        Return the integrated gain up to the longitudinal coordinate z.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinate z

        Returns
        -------
        numpy.ndarray
            Integrated gain up to the longitudinal coordinate z
        '''
        return self._gainz.intshape(z)

    def rho(self):
        '''
        Return the real part of the lineshape at the normalized detuning Omega.

        Returns
        -------
        numpy.float64
            Real part of the lineshape at the normalized detuning Omega
        '''
        return self._lineshape.rho()

    def iota(self):
        '''
        Return the imaginary part of the lineshape at the normalized
        detuning Omega.

        Returns
        -------
        numpy.float64
            Imaginary part of the lineshape at the normalized detuning Omega
        '''
        return self._lineshape.iota()

    def ls_annotation(self) -> str:
        '''
        Return the annotation for the lineshape object.

        Returns
        -------
        str
            String containing the annotation for the lineshape object
        '''
        return self._lineshape.annotation()

    def plot_gain(
        self,
        z: npt.NDArray[np.float64],
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None
    ):
        '''
        Plot the gain over the longitudinal coordinates z.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z; must be within the range [0, 1/kappa]
        show : bool, optional
            Flag to display the plot (default is True)
        savepath : str, optional
            Path to save the plot (default is None)
        filename : str, optional
            Name of the file to save the plot (default is None)
        '''
        assert np.min(z) >= self._z_min, (
            "Minimum z = {}; must be >= {}."
            .format(np.min(z), self._z_min)
        )
        assert np.max(z) <= self._z_max, (
            "Maximum z = {}; must be <= {}."
            .format(np.max(z), self._z_max)
        )
        self._gainz.plot_shape(z, show, savepath, filename)

    def plot_intgain(self, z: npt.NDArray[np.float64], show: bool = True,
                     savepath: str | None = None, filename: str | None = None):
        '''
        Plot the integrated gain over the longitudinal coordinates z.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z; must be within the range [0, 1/kappa]
        show : bool, optional
            Flag to display the plot (default is True)
        savepath : str, optional
            Path to save the plot (default is None)
        filename : str, optional
            Name of the file to save the plot (default is None)
        '''
        assert np.min(z) >= self._z_min, (
            "Minimum z = {}; must be >= {}."
            .format(np.min(z), self._z_min)
        )
        assert np.max(z) <= self._z_max, (
            "Maximum z = {}; must be <= {}."
            .format(np.max(z), self._z_max)
        )
        self._gainz.plot_intshape(z, show, savepath, filename)


def GainConstLorentzian(
    name: str,
    gbar_0: float,
    omega: float,
    config: Config
):
    '''
    Create a Gain object with a constant shape and a Lorentzian lineshape.

    Parameters
    ----------
    name : str
        Name of the Gain object
    gbar_0 : numpy.float64
        Integrated unsaturated gain in the amplifier or laser
    omega : numpy.float64
        Normalized detuning Omega
    config : Config
        Configuration of the laser system
    '''
    params_shape: dict[str, object] = {
        'name': 'Constant Shape',
        'value': 1,
        'norm': 1
    }
    params_lineshape: dict[str, object] = {
        'name': 'Lorentzian Lineshape',
        'omega': omega
    }
    params_gain: dict[str, object] = {'name': name, 'gbar_0': gbar_0}

    shape = Const(params_shape)
    lineshape = Lorentzian(params_lineshape)

    return Gain(params_gain, config, shape, lineshape)


def GainTrapLorentzian(
    name: str,
    gbar_0: float,
    z_1: float,
    z_2: float,
    delta: float,
    omega: float,
    config: Config
):
    '''
    Create a Gain object with a trapezoidal shape and a Lorentzian lineshape.

    Parameters
    ----------
    name : str
        Name of the Gain object
    gbar_0 : numpy.float64
        Integrated unsaturated gain in the amplifier or laser
    z_1 : numpy.float64
        Start of the trapezoidal gain profile (normalized by the length of
        the amplifier)
    z_2 : numpy.float64
        End of the trapezoidal gain profile (normalized by the length of
        the amplifier)
    delta : numpy.float64
        Slope of the trapezoidal gain profile
    omega : numpy.float64
        Normalized detuning Omega
    config : Config
        Configuration of the laser system
    '''
    norm = 1 / (config.kappa() * (z_2 - z_1))
    params_shape: dict[str, object] = {
        'name': 'Constant Shape',
        'value': gbar_0,
        'norm': norm,
        'x_1': z_1,
        'x_2': z_2,
        'delta_1': delta,
        'delta_2': delta
    }
    params_lineshape: dict[str, object] = {
        'name': 'Lorentzian Lineshape',
        'omega': omega
    }
    params_gain: dict[str, object] = {'name': name, 'gbar_0': gbar_0}

    shape = TrapAcute(params_shape)
    lineshape = Lorentzian(params_lineshape)

    return Gain(params_gain, config, shape, lineshape)


class LaserAmplifierCW(BaseClass):
    '''
    Class representing a continuous-wave laser amplifier with a
    Lorentzian lineshape.

    Parameters
    ----------
    params : dict
        Dictionary containing the key-value pairs needed to initialize
        a LaserAmplifierCW object; the minimum required keys are:
            'name' : str
                Name of the object
            'alpha_0' : numpy.float64
                Absorption coefficient times amplifier length
            'gbar_0' : numpy.float64
                Integrated single-pass unsaturated gain
            'omega' : numpy.float64
                Normalized detuning Omega
        Additional key-value pairs can be included in the dictionary to set
        the parameters of the LaserAmplifierCW object

    Methods
    -------
    solve(z, i_0)
        Analytically solve for the intensity within the amplifier
    integrate(z, i_0)
        Numerically solve for the intensity within the amplifier
    get_igz()
        Return the intensities, the saturated gains, and the longitudinal
        coordinates as a tuple
    plot_intensity(show=True, savepath=None, filename=None)
        Plot the intensity within the amplifier
    plot_gain(show=True, savepath=None, filename=None)
        Plot the saturated gain of the amplifier
    '''

    _alpha_0: float  # Absorption coefficient times amplifier length
    _gbar_0: float  # Integrated single-pass unsaturated gain
    _omega: float  # Normalized detuning Omega
    _i_0: npt.NDArray[np.float64]  # Initial intensities at z=0
    _iz: npt.NDArray[np.float64]   # Intensities at all z
    _z: npt.NDArray[np.float64]    # Longitudinal coordinates z

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        '''
        self._specs = {
            'name': {'units': ''},      # name of the object
            'alpha_0': {'units': ''},   # absorption coefficient times
                                        # amplifier length
            'gbar_0': {'units': ''},    # integrated single-pass
                                        # unsaturated gain
            'omega': {'units': ''}      # normalized detuning Omega
        }

    def _di_dz(
        self,
        z: npt.NDArray[np.float64],
        iz: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Return the derivative of the intensity with respect to the longitudinal
        coordinate z for a Lorentzian lineshape.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinate z (normalized by the length of
            the amplifier)
        iz : npt.NDArray[np.float64]
            Intensity at the longitudinal coordinate z

        Returns
        -------
        numpy.ndarray
            Derivative of the intensity with respect to the longitudinal
            coordinate z
        '''
        gain_term = self._gbar_0 * iz / (1 + self._omega**2 + iz)
        loss_term = self._alpha_0 * iz
        return gain_term - loss_term

    def _gain(self) -> npt.NDArray[np.float64]:
        '''
        Return the saturated gain of the amplifier at the longitudinal
        coordinates z (through the values of the intensities at z) for a
        Lorentzian lineshape.

        Returns
        -------
        numpy.ndarray
            Saturated gain of the amplifier at the longitudinal coordinates z
        '''
        numerator = self._gbar_0 * (1 + self._omega**2)
        denominator = 1 + self._omega**2 + self._iz
        return numerator / denominator

    def get_igz(
        self
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64]
    ]:
        '''
        Return the intensities, the saturated gains, and the longitudinal
        coordinates as a tuple.
        '''
        return self._iz, self._gz, self._z

    def solve(self, z: npt.NDArray[np.float64], i_0: npt.NDArray[np.float64]):
        '''
        Analytically solve for the intensity within the amplifier.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z (normalized by the length
            of the amplifier)
        i_0 : numpy.ndarray
            Initial intensities at the longitudinal coordinate z=0
        '''
        def funcz(iz: float, z: float, i_0: float) -> float:
            a0 = self._alpha_0
            g0 = self._gbar_0
            w2 = 1 + self._omega**2
            if a0 < 10 * np.finfo(float).eps:
                return np.log(iz / i_0) + ((iz - i_0) - g0 * z) / w2
            else:
                num = g0 - a0 * (w2 + iz)
                denom = g0 - a0 * (w2 + i_0)
                log_term = np.log(num / denom)
                return (
                    w2 * np.log(iz / i_0)
                    - (g0 / a0) * log_term
                    - (g0 - w2 * a0) * z
                )

        self._i_0 = setarr(i_0)
        iz = np.zeros((len(self._i_0), len(z)))
        iz[:, 0] = self._i_0

        for m in range(len(self._i_0)):
            i0 = self._i_0[m]
            for n in range(len(z)-1):
                args = (z[n+1], i0)
                a = iz[m, n]
                b = (
                    iz[m, n]
                    + self._di_dz(z[n], iz[m, n])
                    * 2
                    * (z[n + 1] - z[n])
                )
                iz[m, n+1] = brentq(funcz, a, b, args)

        self._z = z
        self._iz = iz
        self._gz = self._gain()

    def integrate(
        self,
        z: npt.NDArray[np.float64],
        i_0: npt.NDArray[np.float64]
    ):
        '''
        Numerically solve for the intensity within the amplifier.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z (normalized by the length
            of the amplifier)
        i_0 : numpy.ndarray
            Initial intensities at the longitudinal coordinate z=0
        '''
        self._i_0 = setarr(i_0)
        sol = solve_ivp(  # type: ignore
            self._di_dz,
            [0, 1],
            self._i_0,
            vectorized=True,
            dense_output=True
        )
        if not sol.success:
            raise RuntimeError(
                "Numerical integration failed: {}".format(sol.message)
            )
        if sol.sol is None:
            raise RuntimeError(
                "Dense output was not generated. "
                "Check solve_ivp parameters and integration success."
            )
        self._z = z
        self._iz = sol.sol(z)
        self._gz = self._gain()

    def plot_intensity(
        self,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None
    ):
        '''
        Plot the intensity within the amplifier.

        Parameters
        ----------
        show : bool, optional
            Flag to display the plot (default is True)
        savepath : str, optional
            Path to save the plot (default is None)
        filename : str, optional
            Name of the file to save the plot (default is None)
        '''
        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        for n in range(self._iz.shape[0]):
            ax.plot(  # type: ignore
                self._z,
                self._iz[n] / self._iz[n][0],
                label=r'$I_0 = {}$'.format(self._iz[n][0])
            )
        ax.set_xlabel(r'$z$', fontdict=font)  # type: ignore
        ax.set_ylabel(  # type: ignore
            r'$G_\mathrm{eff}(z) \equiv I(z)/I_0$', fontdict=font
        )
        ax.set_xlim(self._z[0], self._z[-1])
        ax.set_ylim(*get_ylim())
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        ax.legend(fontsize=labelsize)  # type: ignore

        figdisp(fig, show, savepath, filename)

    def plot_gain(self, show: bool = True,
                  savepath: str | None = None, filename: str | None = None):
        '''
        Plot the saturated gain of the amplifier.

        Parameters
        ----------
        show : bool, optional
            Flag to display the plot (default is True)
        savepath : str, optional
            Path to save the plot (default is None)
        filename : str, optional
            Name of the file to save the plot (default is None)
        '''
        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        for n in range(self._gz.shape[0]):
            label_str = r'$I_0 = {}$'.format(str(self._iz[n][0]))
            ax.plot(self._z, self._gz[n], label=label_str)  # type: ignore
        ax.set_xlabel(r'$z$', fontdict=font)  # type: ignore
        ax.set_ylabel(r'$G(z)$', fontdict=font)  # type: ignore
        ax.set_xlim(self._z[0], self._z[-1])
        ax.set_ylim(0.0, get_ylim()[1])
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        ax.legend(loc='upper right', fontsize=labelsize)  # type: ignore

        figdisp(fig, show, savepath, filename)


class LaserCW(BaseClass):
    '''
    Virtual base class for continuous-wave laser objects.

    Parameters
    ----------
    params : dict
        Dictionary containing the key-value pairs needed to initialize a
        LaserCW object; the minimum required keys are:
            'name' : str
                Name of the object
            'r_1' : numpy.float64
                Reflectivity of the first mirror
            'a_1' : numpy.float64
                Absorptivity of the first mirror
            'r_2' : numpy.float64
                Reflectivity of the second mirror
            'a_2' : numpy.float64
                Absorptivity of the second mirror
        Additional key-value pairs can be included in the dictionary to set
        the parameters of objects derived from the LaserCW class

    Methods
    -------
    set_params(params)
        Set the parameters of the LaserCW object
    check_config(config)
        Check that the configurations of the gain and loss objects are
        identical to that of the LaserCW object
    threshold()
        Return the threshold gain
    kappa()
        Return the effective saturation parameter
    intensity_0()
        Return an approximate value for the intensity at z = 0
    below_threshold()
        Return a flag indicating whether the gain is below the threshold
    c2()
        Return the square of the spatial quasi-normal nmode
        normalization constant
    '''
    _name: str   # Name of the object
    _r_1: float  # Reflectivity of the first mirror
    _a_1: float  # Absorptivity of the first mirror
    _r_2: float  # Reflectivity of the second mirror
    _a_2: float  # Absorptivity of the second mirror
    _gain: Gain  # The gain profile in space and frequency
    _loss: Gain  # The loss profile in space and frequency
    _z_min: float  # Minimum value of the normalized longitudinal coordinate z
    _z_max: float  # Maximum value of the normalized longitudinal coordinate z

    def __init__(self, params: dict[str, object], gain: Gain, loss: Gain):
        '''
        Initialize a LaserCW object.

        Parameters
        ----------
        params : dict
            Dictionary containing the following key-value pairs needed to
            initialize a LaserCW object; the minimum required keys are:
                'name' : str
                    Name of the object
                'r_1' : numpy.float64
                    Reflectivity of the first mirror
                'a_1' : numpy.float64
                    Absorptivity of the first mirror
                'r_2' : numpy.float64
                    Reflectivity of the second mirror
                'a_2' : numpy.float64
                    Absorptivity of the second mirror
            Additional key-value pairs can be included in the dictionary to
            set the parameters of objects derived from the LaserCW class
        gain : Gain
            Gain object representing the gain profile in space and frequency
        loss : Gain
            Gain object representing the loss profile in space and frequency
        '''
        self._set_specs()
        self._specs['z_min'] = {'units': ''}
        self._specs['z_max'] = {'units': ''}
        self._specs['gbar_th'] = {'units': ''}

        params['z_min'] = self._config().zlims()[0]
        params['z_max'] = self._config().zlims()[1]
        params['gbar_th'] = 0.0

        self._gain = deepcopy(gain)
        self._loss = deepcopy(loss)

        # Set mirror and absorptivity attributes from params
        self._r_1 = convert_to_float(params.get('r_1'))
        self._a_1 = convert_to_float(params.get('a_1'))
        self._r_2 = convert_to_float(params.get('r_2'))
        self._a_2 = convert_to_float(params.get('a_2'))

        self.set_params(params)
        self._gbar_th = self._threshold()

        self._check_vars()

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        '''
        self._specs = {'name': {'units': ''},  # name of the object
                       'r_1': {'units': ''},   # reflectivity of the mirror 1
                       'a_1': {'units': ''},   # absorptivity of the mirror 1
                       'r_2': {'units': ''},   # reflectivity of the mirror 2
                       'a_2': {'units': ''}    # absorptivity of the mirror 2
                       }

    def _config(self) -> Config:
        '''
        Return the configuration of the laser system.
        '''
        raise NotImplementedError

    def _check_vars(self):
        '''
        Check the values of input parameters/variables.

        Raises
        ------
        AssertionError
            If the values are invalid. The requirements are:
                0.0 < r_1 < 1.0
                0.0 <= a_1 < 1.0
                r_1 + a_1 < 1.0
                0.0 < r_2 <= 1.0 (SWL) or r_2 = 1.0 (URL)
                0.0 <= a_2 <= 1.0 (SWL) or a_2 = 0.0 (URL)
                r_2 + a_2 <= 1.0
        '''
        self._gain.check_config(self._config())
        self._loss.check_config(self._config())

        assert self._r_1 > 0.0, "r_1 = {}; must be > 0.".format(self._r_1)
        assert self._r_1 < 1.0, "r_1 = {}; must be < 1.".format(self._r_1)
        assert self._a_1 >= 0.0, "a_1 = {}; must be >= 0.".format(self._a_1)
        assert self._r_1 + self._a_1 < 1.0, (
            "r_1 + a_1 = {}; must be < 1."
            .format(self._r_1 + self._a_1)
        )
        assert self._r_2 <= 1.0, (
            "r_2 = {}; must be <= 1."
            .format(self._r_2)
        )

        if self._config().name == 'URL':
            assert self._r_2 == 1.0, "r_2 = {}; must be 1.".format(self._r_2)
            assert self._a_2 == 0.0, "a_2 = {}; must be 0.".format(self._a_2)
        else:
            assert self._r_2 > 0.0, "r_2 = {}; must be > 0.".format(self._r_2)
            assert self._r_2 <= 1.0, (
                "r_2 = {}; must be <= 1."
                .format(self._r_2)
            )
            assert self._a_2 >= 0.0, (
                "a_2 = {}; must be >= 0."
                .format(self._a_2)
            )
            assert self._r_2 + self._a_2 <= 1.0, (
                "r_2 + a_2 = {}; must be <= 1."
                .format(self._r_2 + self._a_2)
            )

    def _threshold(self):
        '''
        Return the threshold gain.

        Returns
        -------
        numpy.float64
            Threshold gain
        '''
        r = self._r_1 * self._r_2
        abar_0 = convert_to_float(self._loss.get_params()['gbar_0'])
        rho = self._gain.rho()

        return np.log(1/(r * np.exp(abar_0))) / rho

    def _kappa(self):
        '''
        Return the effective saturation parameter.

        Returns
        -------
        numpy.float64
            Effective saturation parameter
        '''
        return self._config().kappa()

    def _intensity_0(self, kappa: float) -> float:
        '''
        Return an approximate value for the intensity at z = 0.

        Parameters
        ----------
        kappa : numpy.float64
            Effective saturation parameter

        Returns
        -------
        numpy.float64
            Approximate value for the intensity at z = 0
        '''
        if self._below_threshold():
            return 0.0

        gbar_0 = self._gain.get_params()['gbar_0']
        gbar_th = self._threshold()
        rho = self._gain.rho()

        return (gbar_0 / gbar_th - 1) / (kappa * rho)

    def _below_threshold(self):
        '''
        Return a flag indicating whether gbar_0 is less than or equal to
        gbar_th.

        Returns
        -------
        bool
            Flag indicating whether gbar_0 is less than or equal to gbar_th
        '''
        gbar_0 = self._gain.get_params()['gbar_0']

        return gbar_0 <= self._threshold()

    def _c2(self):
        '''
        Return the square of the spatial quasi-normal nmode normalization
        constant.

        Returns
        -------
        numpy.float64
            Square of the spatial quasi-normal nmode normalization constant
        '''
        r_1 = self._r_1
        r_2 = self._r_2

        return (
            (r_1 * np.sqrt(r_2) * np.log(1.0 / (r_1 * r_2)))
            / (
                (np.sqrt(r_1) + np.sqrt(r_2))
                * (1 - np.sqrt(r_1 * r_2))
            )
        )

    def _kz(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        '''
        Return the exponential of the product of the position and the spatial
        quasi-normal nmode propagation vector.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z (normalized by the laser cavity
            round-trip length)

        Returns
        -------
        numpy.ndarray
            Exponential of the product of the position and the spatial
            quasi-normal mode propagation vector
        '''
        r = self._r_1 * self._r_2
        abar_0 = convert_to_float(self._loss.get_params()['gbar_0'])
        gbar_0 = convert_to_float(self._gain.get_params()['gbar_0'])
        beta = np.log(1 / (r * np.exp(abar_0))) / gbar_0

        return np.exp(beta * self._gain.intgain(z) + self._loss.intgain(z))

    def _grmg(
        self,
        gbar_0: npt.NDArray[np.float64],
        npts: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        '''
        Return the meshgrid of the gain and the reflectivity; columns
        corresponding to the gain, and rows correspond to the reflectivity
        _above threshold_.

        Parameters
        ----------
        gbar_0 : numpy.ndarray
            Round-trip ntegrated unsaturated gain in the laser
        npts : int
            Number of points in the meshgrid corresponding to the reflectivity
            _above threshold_

        Returns
        -------
        numpy.ndarray
            Meshgrid of the gain and the reflectivity
        '''
        g_mg = np.tile(gbar_0, (npts, 1))

        r_mg = np.zeros_like(g_mg)
        for n in range(len(gbar_0)):
            abar_0 = convert_to_float(self._loss.get_params()['gbar_0'])
            exp_arg = -float(gbar_0[n]) - abar_0
            r_min = (
                self._gain.rho() * np.exp(exp_arg)
                + np.finfo(float).eps
            )
            r_max = 1.0 - self._a_1 - 10 * np.finfo(float).eps
            r = np.linspace(
                r_min, r_max, npts, endpoint=True, dtype=np.float64
            )
            r_mg[:, n] = r

        return (r_mg, g_mg)

    def _di_dz(
        self,
        z: npt.NDArray[np.float64],
        iz: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Return the derivative of the intensity with respect to the longitudinal
        coordinate z.

        Parameters
        ----------
        z : numpy.float64
            Longitudinal coordinate z (normalized by the round-trip length of
            the laser cavity)
        iz : numpy.float64
            Intensity at the longitudinal coordinate z

        Returns
        -------
        numpy.float64
            Derivative of the intensity with respect to the longitudinal
            coordinate z
        '''
        raise NotImplementedError

    def _bc(
        self,
        ia: npt.NDArray[np.float64],
        ib: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Return the boundary conditions for the intensity at the longitudinal
        coordinates z = 0 and z = 1.

        Parameters
        ----------
        ia : numpy.float64
            Intensity at the longitudinal coordinate z = 0
        ib : numpy.float64
            Intensity at the longitudinal coordinate z = 1

        Returns
        -------
        numpy.ndarray
            Boundary conditions for the intensity at the longitudinal
            coordinates z = 0 and z = 1
        '''
        raise NotImplementedError

    def _exact_io(
        self,
        g_mg: npt.NDArray[np.float64],
        r_mg: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], str]:
        '''
        Return the exact intensity at the longitudinal coordinates z = 0
        and z = 1.

        Parameters
        ----------
        g_mg : numpy.ndarray
            Meshgrid of the gain
        r_mg : numpy.ndarray
            Meshgrid of the reflectivity _above threshold_

        Returns
        -------
        tuple
            Exact intensity at the longitudinal coordinates z = 0 and z = 1
        '''
        raise NotImplementedError

    def _funcr(
        self,
        r: npt.NDArray[np.float64],
        g: npt.NDArray[np.float64]
    ) -> float:
        '''
        Return the function to be minimized to find the exact value of the
        optimum reflectivity.

        Parameters
        ----------
        r : float or numpy.ndarray
            Reflectivity of the output coupler
        g : float or numpy.ndarray
            Integrated unsaturated round-trip gain in the laser
        '''
        raise NotImplementedError

    # def set_params(self, params: dict[str, object]):
    #     '''
    #     Set the parameters of the LaserCW object.

    #     Parameters
    #     ----------
    #     params : dict
    #         Dictionary containing the key-value pairs needed to set the
    #         parameters of the LaserCW object and its constituent objects
    #         (gain and loss).
    #     '''
    #     if (
    #         'name' in params
    #         and params['name'] == self._gain.get_params()['name']
    #     ):
    #         self._gain.set_params(params['params'])
    #     elif (
    #         'name' in params
    #         and params['name'] == self._loss.get_params()['name']
    #     ):
    #         self._loss.set_params(params['params'])
    #     else:
    #         BaseClass.set_params(self, params)

    def set_params(self, params: dict[str, object]):
        '''
        Set the parameters of the Gain object.

        Parameters
        ----------
        params : dict
            Dictionary containing the parameters to set; the keys must be
            present in the _specs dictionaries of the LaserCW, Gain, and/or
            Loss objects. If the name key is present in the params dictionary,
            and it matches the name of either the Gain or Loss object,
            the parameters are set in the corresponding object. Otherwise, the
            parameters are set in the attributes of the LaserCW object

        Note
        ----
        The names of the Gain and Loss objects can't be changed.
        '''
        if (
            'name' in params
            and params['name'] == self._gain.get_params()['name']
        ):
            del params['name']
            self._gain.set_params(params)
        elif (
            'name' in params
            and params['name'] == self._loss.get_params()['name']
        ):
            del params['name']
            self._loss.set_params(params)
        else:
            BaseClass.set_params(self, params)

    def approx_iz(
        self,
        z: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        '''
        Return the approximate intensity at the longitudinal coordinates z.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z (normalized by the round-trip length of
            the laser cavity)

        Returns
        -------
        tuple
            Approximate forward and backward intensities at the longitudinal
            coordinates z
        '''
        if self._below_threshold():
            ip_z = np.zeros_like(z)
            im_z = np.zeros_like(z)
        else:
            c2 = self._c2()
            kz = self._kz(z)
            up_0 = c2 * kz
            um_0 = c2 / (self._r_1 * kz)

            i_0 = self._intensity_0(self._kappa())

            ip_z = i_0 * up_0
            im_z = i_0 * um_0

        return ip_z, im_z

    def approx_io(
        self,
        g: npt.NDArray[np.float64],
        r: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Return the approximate output intensity.

        Parameters
        ----------
        g : numpy.ndarray
            Integrated unsaturated round-trip gain in the laser
        r : numpy.ndarray
            Reflectivity of the output coupler

        Returns
        -------
        numpy.ndarray
            Approximate output intensity
        '''
        store = Storage(gbar_0=self._gain.get_params()['gbar_0'],
                        r_1=self._r_1, r_2=self._r_2, a_2=self._a_2)
        self.set_params({'r_2': 1.0, 'a_2': 0.0})

        i_out = np.zeros_like(g)
        for m in np.ndindex(g.shape):
            self._gain.set_params({'gbar_0': g[m]})
            self.set_params({'r_1': r[m]})
            i_0 = self._intensity_0(self._kappa())
            c2 = self._c2()

            i_out[m] = ((1.0 - self._a_1 - r[m]) * c2 / r[m]) * i_0

        self.set_params({
            'r_1': store.r_1,  # type: ignore
            'r_2': store.r_2,  # type: ignore
            'a_2': store.a_2  # type: ignore
        })
        self._gain.set_params({'gbar_0': store.gbar_0})  # type: ignore

        i_out[i_out < 0] = -1

        return i_out

    def solve_io(
        self,
        g: npt.NDArray[np.float64],
        r: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Return an analytic solution for the output intensity.

        Parameters
        ----------
        g : numpy.ndarray
            Integrated unsaturated round-trip gain in the laser
        r : numpy.ndarray
            Reflectivity of the output coupler

        Returns
        -------
        numpy.ndarray
            Exact solution for the output intensity
        '''
        raise NotImplementedError

    def solve_iz(
        self,
        z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64] | tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        '''
        Return an exact solution for the intensity at the longitudinal
        coordinates z.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z (normalized by the round-trip length of
            the laser cavity)

        Returns
        -------
        numpy.ndarray
            Exact solution for the intensity at the longitudinal coordinates z
        '''
        raise NotImplementedError

    def integrate_io(
        self,
        g: npt.NDArray[np.float64],
        r: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Return a numerical solution for the output intensity.

        Parameters
        ----------
        gbar_0 : numpy.ndarray
            Integrated unsaturated round-trip gain in the laser
        r : numpy.ndarray
            Reflectivity of the output coupler

        Returns
        -------
        numpy.ndarray
            Numerical solution for the output intensity
        '''
        raise NotImplementedError

    from typing import Literal

    def integrate_iz(
        self,
        z: npt.NDArray[np.float64],
        verbose: 'Literal[0, 1, 2]' = 0
    ):
        '''
        Return a numerical solution for the intensity at the longitudinal
        coordinates z.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z (normalized by the round-trip length of
            the laser cavity)
        verbose : Literal[0, 1, 2], optional
            Flag to display the progress of the numerical solution
            (default is 0)

        Returns
        -------
        scipy.integrate.solve_bvp bunch object
            Solution for the intensity at the longitudinal coordinates z
        '''
        iz = np.zeros((2, z.size))
        iz[0], iz[1] = self.approx_iz(z)

        sol = solve_bvp(self._di_dz, self._bc, z, iz, verbose=verbose)
        if verbose != 0:
            print("\n")
        return sol

    def approx_opt(
        self,
        g_max: float,
        npts: int = 101
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
               npt.NDArray[np.float64]]:
        '''
        Return the approximate optimum reflectivity and intensity at the
        output coupler.

        Parameters
        ----------
        g_max : numpy.float64
            Maximum integrated unsaturated round-trip gain in the laser
        npts : int, optional
            Number of points in the gain array  (default is 101)

        Returns
        -------
        tuple
            Gain array, approximate optimum reflectivity, and approximate
            optimum output intensity
        '''
        abar_0 = convert_to_float(self._loss.get_params()['gbar_0'])

        gp2 = np.exp(abar_0)
        loss = -np.log((1 - self._a_1) * gp2)
        g_min = loss/self._gain.rho() + np.finfo(float).eps
        gbar_0 = np.linspace(
            g_min, g_max, npts, endpoint=True, dtype=np.float64
        )

        r_opt = np.exp(-np.sqrt(self._gain.rho() * gbar_0 * loss)) / gp2
        i_opt = self.approx_io(gbar_0, r_opt)

        return gbar_0, r_opt, i_opt

    def exact_opt(
        self,
        g_max: float,
        npts: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        '''
        Return the exact optimum reflectivity and optimum output intensity.

        Parameters
        ----------
        g_max : numpy.float64
            Maximum integrated unsaturated round-trip gain in the laser
        npts : int
            Number of points in the gain array

        Returns
        -------
        tuple
            Exact optimum reflectivity and exact optimum output intensity
        '''
        gbar_0, r_x_opt, i_x_opt = self.approx_opt(g_max, npts)

        r_e_opt = np.zeros_like(r_x_opt)
        i_e_opt = np.zeros_like(i_x_opt)

        for m in np.ndindex(gbar_0.shape):
            sol = fmin(
                self._funcr,
                r_x_opt[m],
                args=(gbar_0[m],),
                full_output=True,
                disp=False
            )
            r_e_opt[m] = float(sol[0])
            i_e_opt[m] = -float(sol[1])

        return r_e_opt, i_e_opt

    def plot_r(
        self,
        gbar_0: npt.NDArray[np.float64],
        npts: int = 101,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None
    ):
        '''
        Plot the output intensity as a function of the reflectivity of the
        output coupler.

        Parameters
        ----------
        gbar_0 : numpy.ndarray
            Integrated unsaturated round-trip gain in the laser
        npts : int, optional
            Number of points to be used in the reflectivity array
            (default is 101)
        show : bool, optional
            Flag to display the plot (default is True)
        savepath : str, optional
            Path to save the plot (default is None)
        filename : str, optional
            Name of the file to save the plot (default is None)
        '''
        r_mg, g_mg = self._grmg(gbar_0, npts)
        i_x = self.approx_io(g_mg, r_mg)
        i_e, label_e = self._exact_io(g_mg, r_mg)

        gbar_0_loss = convert_to_float(self._loss.get_params()['gbar_0'])
        annotation = (
            r'$\overline{{\alpha}}_0 = {:.2f}$'.format(abs(gbar_0_loss)) + '\n'
            + r'$A = {}$'.format(self._a_1) + '\n'
            + self._gain.ls_annotation()
        )

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        for n in range(gbar_0.size):
            label_str = (
                r'$\overline{{G}}_0 = {}$'.format(gbar_0[n]) + str(label_e)
            )
            ax.plot(r_mg[:, n], i_e[:, n], label=label_str)  # type: ignore
            ax.plot(  # type: ignore
                r_mg[:, n],
                i_x[:, n],
                '--',
                label=(
                    r'$\overline{{G}}_0 = {}$: approximate'
                    .format(gbar_0[n])
                )
            )
        ax.set_xlabel(r'$R$', fontdict=font)  # type: ignore
        ax.set_ylabel(r'$I_\mathrm{out}$', fontdict=font)  # type: ignore
        ax.set_xlim(0, 1)
        ax.set_ylim(np.finfo(float).eps, get_ylim()[1])  # type: ignore
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        ax.legend(loc='upper left', fontsize=labelsize)  # type: ignore
        ob = offsetbox.AnchoredText(
            annotation,
            loc='upper right',
            pad=0,
            borderpad=0.65,
            prop=dict(size=fontsize)
        )
        ob.patch.set(  # type: ignore
            boxstyle='round',
            edgecolor='#D7D7D7',
            facecolor='white',
            alpha=0.75
        )
        ax.add_artist(ob)

        figdisp(fig, show, savepath, filename)

    def plot_opt(
        self,
        g_max: float,
        npts: int = 101,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None
    ):
        '''
        Plot the optimum reflectivity and output intensity as a function of
        the integrated unsaturated round-trip gain in the laser.

        Parameters
        ----------
        g_max : numpy.float64
            Maximum integrated unsaturated round-trip gain in the laser
        npts : int, optional
            Number of points to be used in the gain array  (default is 101)
        show : bool, optional
            Flag to display the plots (default is True)
        savepath : str, optional
            Path to save the plots (default is None)
        filename : str, optional
            Name of the file to save the plots (default is None)
        '''
        gbar_0, r_x_opt, i_x_opt = self.approx_opt(g_max, npts)
        r_e_opt, i_e_opt = self.exact_opt(g_max, npts)

        abar_0 = convert_to_float(self._loss.get_params()['gbar_0'])
        annotation = (
            r'$\overline{{\alpha}}_0 = {:.2f}$'.format(abs(abar_0)) + '\n'
            + r'$A = {}$'.format(self._a_1) + '\n'
            + self._gain.ls_annotation()
        )

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        ax.plot(gbar_0, r_e_opt, label='exact')  # type: ignore
        ax.plot(gbar_0, r_x_opt, '--', label='approx')  # type: ignore
        ax.set_xlabel(r'$\overline{{G}}_0$', fontdict=font)  # type: ignore
        ax.set_ylabel(r'$R_\mathrm{opt}$', fontdict=font)  # type: ignore
        ax.set_xlim(0, g_max)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        ax.legend(fontsize=labelsize)  # type: ignore
        ob = offsetbox.AnchoredText(
            annotation,
            loc='lower left',
            pad=0,
            borderpad=0.65,
            prop=dict(size=fontsize)
        )
        ob.patch.set(  # type: ignore
            boxstyle='round',
            edgecolor='#D7D7D7',
            facecolor='white',
            alpha=0.75
        )
        ax.add_artist(ob)

        filepath = file_path(savepath, filename)
        if filepath is None:
            pass
        else:
            idx = filepath.index('.pdf')
            filepath_r = filepath[:idx] + '_r' + filepath[idx:]
            fig.savefig(filepath_r, bbox_inches='tight')  # type: ignore
            print("Saved {0}\n".format(filepath_r))
        if show:
            plt.show()  # type: ignore
        else:
            plt.close()  # type: ignore

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        ax.plot(gbar_0, i_e_opt, label='exact')  # type: ignore
        ax.plot(gbar_0, i_x_opt, '--', label='approx')  # type: ignore
        ax.set_xlabel(r'$\overline{{G}}_0$', fontdict=font)  # type: ignore
        ax.set_ylabel(r'$I_\mathrm{opt}$', fontdict=font)  # type: ignore
        ax.set_xlim(0, g_max)
        ax.set_ylim(0, get_ylim()[1])
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        ax.legend(fontsize=labelsize)  # type: ignore
        ob = offsetbox.AnchoredText(
            annotation,
            loc='lower right',
            pad=0,
            borderpad=0.65,
            prop=dict(size=fontsize)
        )
        ob.patch.set(  # type: ignore
            boxstyle='round',
            edgecolor='#D7D7D7',
            facecolor='white',
            alpha=0.75
        )
        ax.add_artist(ob)

        filepath = file_path(savepath, filename)
        if filepath is None:
            pass
        else:
            idx = filepath.index('.pdf')
            filepath_i = filepath[:idx] + '_i' + filepath[idx:]
            fig.savefig(filepath_i, bbox_inches='tight')  # type: ignore
            print("Saved {0}\n".format(filepath_i))
        if show:
            plt.show()  # type: ignore
        else:
            plt.close()  # type: ignore


class LaserCWURL(LaserCW):
    '''
    Model a continuous-wave (CW) unidirectional ring laser derived from
    the virtual base class LaserCW
    '''

    def _config(self):
        '''
        Return the configuration of the laser: URL

        Returns
        -------
        Config.URL
        '''
        return Config.URL

    def _di_dz(
        self,
        z: npt.NDArray[np.float64],
        iz: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Return the derivative of the intensity with respect to the longitudinal
        coordinate z.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z (normalized by the round-trip length of
            the laser cavity)
        iz : numpy.ndarray
            Intensity at the longitudinal coordinates z

        Returns
        -------
        numpy.ndarray
            Derivative of the intensity with respect to the longitudinal
            coordinate z
        '''
        dip_dz = (
            self._gain.gain(z) * iz[0] / (1 / self._gain.rho() + iz[0])
            + self._loss.gain(z) * iz[0]
        )
        dim_dz = (
            -self._gain.gain(z) * iz[1] / (1 / self._gain.rho() + iz[1])
            - self._loss.gain(z) * iz[1]
        )
        return np.vstack((dip_dz, dim_dz))

    def _bc(
        self,
        ia: npt.NDArray[np.float64],
        ib: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Return the boundary conditions for the forward and backward intensities
        at the output coupler.

        Parameters
        ----------
        ia : numpy.ndarray
            Forward intensity at the output coupler
        ib : numpy.ndarray
            Backward intensity at the output coupler

        Returns
        -------
        numpy.ndarray
            Boundary conditions for the forward and backward intensities
            at the output coupler
        '''
        return np.array([ia[0] - self._r_1 * ib[0], ib[1] - self._r_1 * ia[1]])

    def _exact_io(
        self,
        g_mg: npt.NDArray[np.float64],
        r_mg: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], str]:
        '''
        Return the exact solution for the intensity at the output coupler.

        Parameters
        ----------
        g_mg : numpy.ndarray
            Integrated unsaturated round-trip gain in the laser
        r_mg : numpy.ndarray
            Reflectivity of the output coupler

        Returns
        -------
        tuple
            Exact solution for the intensity at the output coupler and label
        '''
        return self.solve_io(g_mg, r_mg), ': solve (brentq)'

    def _funcr(
        self,
        r: npt.NDArray[np.float64],
        g: npt.NDArray[np.float64]
    ) -> float:
        '''
        Return the function to minimize for the exact solution of the optimum
        reflectivity.

        Parameters
        ----------
        r : numpy.float64
            Reflectivity of the output coupler
        g : numpy.float64
            Integrated unsaturated round-trip gain in the laser

        Returns
        -------
        numpy.float64
            Function to minimize for the exact solution of the optimum
            reflectivity
        '''
        return -self.solve_io(np.array([g]), r)[0]

    def solve_iz(
        self,
        z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Return the exact solution for the intensity at the longitudinal
        coordinates z.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z (normalized by the round-trip length of
            the laser cavity)

        Returns
        -------
        numpy.ndarray
            Exact solution for the intensity at the longitudinal coordinates z
        '''
        def funcz(
            iz: float,
            z: float,
            i0: float,
            gbar_0: float,
            abar_0: float,
            w2: float
        ) -> float:
            if abar_0 < np.finfo(float).eps:
                return np.log(iz/i0) + ((iz - i0) - gbar_0 * z) / w2
            else:
                return (
                    w2 * np.log(iz / i0)
                    - (gbar_0 / abar_0)
                    * np.log(
                        (gbar_0 - abar_0 * (w2 + iz))
                        / (gbar_0 - abar_0 * (w2 + i0))
                    )
                    - (gbar_0 - w2 * abar_0) * z
                )

        iz = np.zeros_like(z)
        if self._below_threshold():
            return iz

        gbar_0 = convert_to_float(self._gain.get_params()['gbar_0'])
        abar_0 = abs(convert_to_float(self._loss.get_params()['gbar_0']))
        w2 = 1/self._gain.rho()
        gbar_th = self._threshold()

        if abar_0 < np.finfo(float).eps:
            i_1 = (gbar_0 - gbar_th) / (1 - self._r_1)
        else:
            x = (abar_0/gbar_0) * (gbar_0 - gbar_th)
            i_1 = (
                (gbar_0 / abar_0 - w2)
                * (1 - np.exp(-x))
                / (1 - self._r_1 * np.exp(-x))
            )
        i_0 = self._r_1 * i_1

        iz[0] = i_0
        iz[-1] = i_1

        for n in range(len(z)-2):
            args = (z[n+1], i_0, gbar_0, abar_0, w2)
            iz[n+1] = brentq(funcz, i_0, i_1, args)

        return iz

    def solve_io(
        self,
        g: npt.NDArray[np.float64],
        r: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Return the exact solution for the output intensity.

        Parameters
        ----------
        g : numpy.ndarray
            Integrated unsaturated round-trip gain in the laser
        r : numpy.ndarray
            Reflectivity of the output coupler

        Returns
        -------
        numpy.ndarray
            Exact solution for the output intensity
        '''
        store = Storage(r_1=self._r_1, r_2=self._r_2, a_2=self._a_2)
        self.set_params({'r_2': 1.0, 'a_2': 0.0})

        i_out = np.zeros_like(g)
        for m in np.ndindex(g.shape):
            self._gain.set_params({'gbar_0': g[m]})
            self.set_params({'r_1': r[m]})
            gbar_th = self._threshold()
            abar_0 = abs(convert_to_float(self._loss.get_params()['gbar_0']))
            w2 = 1/self._gain.rho()
            if abar_0 < np.finfo(float).eps:
                i_1 = (g[m] - gbar_th) / (1 - r[m])
            else:
                x = (abar_0/g[m]) * (g[m] - gbar_th)
                i_1 = (
                    (g[m] / abar_0 - w2)
                    * (1 - np.exp(-x))
                    / (1 - r[m] * np.exp(-x))
                )
            i_out[m] = (1 - self._a_1 - r[m]) * i_1

        self.set_params({
            'r_1': store.r_1,  # type: ignore
            'r_2': store.r_2,  # type: ignore
            'a_2': store.a_2  # type: ignore
        })

        i_out[i_out < 0] = -1

        return i_out

    def plot_z(
        self,
        npts: int = 101,
        bpts: int = 15,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None
    ):
        '''
        Plot the intensity as a function of the longitudinal coordinate z.

        Parameters
        ----------
        npts : int, optional
            Number of points to be used in the z array (default is 101)
        bpts : int, optional
            Initial number of nodes to be used to solve the boundary value
            problem (default is 15)
        show : bool, optional
            Flag to display the plots (default is True)
        savepath : str, optional
            Path to save the plots (default is None)
        filename : str, optional
            Name of the file to save the plots (default is None)
        '''
        z_bvp = np.linspace(0, 1.0, bpts, endpoint=True)
        bvp = self.integrate_iz(z_bvp, verbose=2)

        z = np.linspace(0, 1.0, npts, endpoint=True)
        ip_i = bvp.sol(z)[0]
        ip_s = self.solve_iz(z)
        ip_x = self.approx_iz(z)[0]

        annotation = (
            r'$\overline{{G}}_0 = {:.{prec}}$'.format(
                convert_to_float(self._gain.get_params()['gbar_0']), prec=3
            ) + '\n'
            + r'$\overline{{\alpha}}_0 = {:.{prec}}$'.format(
                abs(
                    convert_to_float(
                        self._loss.get_params()['gbar_0']
                    )
                ),
                prec=2
            ) + '\n'
            + r'$R = {}$'.format(self._r_1) + '\n'
            + self._gain.ls_annotation()
        )

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        ax.plot(z, ip_i, label='integrate (bvp)')  # type: ignore
        ax.plot(z, ip_s, '--', label='solve (brentq)')  # type: ignore
        ax.plot(z, ip_x, label='approximate')  # type: ignore
        ax.set_xlabel(r'$z$', fontdict=font)  # type: ignore
        ax.set_ylabel(r'$I(z)$', fontdict=font)  # type: ignore
        ax.set_xlim(z[0], z[-1])  # type: ignore
        ax.set_ylim(*get_ylim())
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        ax.legend(fontsize=labelsize)  # type: ignore
        ob = offsetbox.AnchoredText(
            annotation,
            loc='lower right',
            pad=0,
            borderpad=0.65,
            prop=dict(size=fontsize)
        )
        ob.patch.set(  # type: ignore
            boxstyle='round',
            edgecolor='#D7D7D7',
            facecolor='white',
            alpha=0.75
        )
        ax.add_artist(ob)

        figdisp(fig, show, savepath, filename)


class LaserCWSWL(LaserCW):
    '''
    Model a continuous-wave (CW) standing-wave laser (based on Rigrod's
    approach) derived from the virtual base class LaserCW
    '''

    def _config(self):
        '''
        Return the configuration of the laser: SWL

        Returns
        -------
        Config.SWL
        '''
        return Config.SWL

    def _phi_rr(self):
        '''
        Return phi_0 computed using Rigrod's SWL model.

        Returns
        -------
        numpy.float64
            phi_0 computed using Rigrod's SWL model
        '''
        kappa = self._config().kappa()
        rho = self._gain.rho()

        gbar_0 = self._gain.get_params()['gbar_0']
        gbar_th = np.log(1/(self._r_1 * self._r_2)) / rho

        i_0 = (gbar_0 / gbar_th - 1) / (kappa * rho)

        return (self._c2() / np.sqrt(self._r_1)) * i_0

    def _di_dz(self, z: npt.NDArray[np.float64], iz: npt.NDArray[np.float64]):
        '''
        Return the derivatives of the forward and backward intensities
        with respect to the longitudinal coordinate z.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z (normalized by the length of the laser
            cavity)
        iz : numpy.ndarray
            Forward and backward intensities at the longitudinal coordinates z

        Returns
        -------
        numpy.ndarray
            Derivatives of the forward and backward intensities with respect
            to the longitudinal coordinate z
        '''
        dip_dz = (
            self._gain.gain(z) * iz[0]
            / (
                1 / self._gain.rho()
                + iz[0]
                + iz[1]
            )
            + self._loss.gain(z) * iz[0]
        )
        dim_dz = (
            -self._gain.gain(z) * iz[1]
            / (1/self._gain.rho() + iz[0] + iz[1])
            - self._loss.gain(z) * iz[1]
        )

        return np.vstack((dip_dz, dim_dz))

    def _bc(
        self,
        ia: npt.NDArray[np.float64],
        ib: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Return the boundary conditions for the forward and backward intensities
        at the two mirrors.

        Parameters
        ----------
        ia : numpy.ndarray
            Forward and backward intensities at the first mirror
        ib : numpy.ndarray
            Forward and backward intensities at the second mirror

        Returns
        -------
        numpy.ndarray
            Boundary conditions for the forward and backward intensities
            at the two mirrors
        '''
        return np.array([ia[0] - self._r_1 * ia[1], ib[1] - self._r_2 * ib[0]])

    def _exact_io(
        self,
        g_mg: npt.NDArray[np.float64],
        r_mg: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], str]:
        '''
        Return the exact solution for the backward intensity (using
        integrate_io) at the output coupler (M1).

        Parameters
        ----------
        g_mg : numpy.ndarray
            Integrated unsaturated round-trip gain in the laser
        r_mg : numpy.ndarray
            Reflectivity of the output coupler

        Returns
        -------
        tuple
            Exact solution for the backward intensity at the output coupler
            and label
        '''
        return (self.integrate_io(g_mg, r_mg), ': integrate (bvp)')

    def _funcr(
        self,
        r: npt.NDArray[np.float64],
        g: npt.NDArray[np.float64]
    ) -> float:
        '''
        Return the function to minimize (using integrate_io) for the exact
        solution of the optimum reflectivity.

        Parameters
        ----------
        r : numpy.float64
            Reflectivity of the output coupler
        g : numpy.float64
            Integrated unsaturated round-trip gain in the laser

        Returns
        -------
        float
            Function to minimize for the exact solution of the optimum
            reflectivity
        '''
        return -self.integrate_io(np.array([g]), r)[0]

    def solve_iz(
        self,
        z: npt.NDArray[np.float64]
    ) -> (
        npt.NDArray[np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ):
        '''
        Return the exact solution for the forward and backward intensities at
        the longitudinal coordinates z.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z (normalized by the length of the laser
            cavity)

        Returns
        -------
        numpy.ndarray
            Exact solution for the forward and backward intensities at the
            longitudinal coordinates z
        '''
        def funcz(
            iz: float,
            z: float,
            i0: float,
            gbar_0: float,
            w2: float,
            phi_rr: float
        ) -> float:
            return (
                w2 * np.log(iz / i0)
                + (iz - i0) * (1 + phi_rr**2 / (iz * i0))
                - gbar_0 * z
            )

        iz = np.zeros_like(z)
        if self._below_threshold():
            return iz, iz

        gbar_0 = self._gain.get_params()['gbar_0']
        w2 = 1/self._gain.rho()
        phi_rr = self._phi_rr()

        i_0 = phi_rr * np.sqrt(self._r_1)
        i_1 = phi_rr / np.sqrt(self._r_2)

        iz[0] = i_0
        iz[-1] = i_1
        for n in range(len(z)-2):
            args = (z[n+1], i_0, gbar_0, w2, phi_rr)
            iz[n+1] = brentq(funcz, i_0, i_1, args)

        return iz, phi_rr**2 / iz

    def integrate_io(
        self,
        g: npt.NDArray[np.float64],
        r: npt.NDArray[np.float64],
        bpts: int = 15
    ) -> npt.NDArray[np.float64]:
        '''
        Return the solution for the output intensity (from integrate_iz) at M1.

        Parameters
        ----------
        g : numpy.ndarray
            Integrated unsaturated round-trip gain in the laser
        r : numpy.ndarray
            Reflectivity of the output coupler
        bpts : int, optional
            Initial number of nodes to be used to solve the boundary value
            problem (default is 15)

        Returns
        -------
        numpy.ndarray
            Solution for the output intensity at M1
        '''
        store = Storage(gbar_0=self._gain.get_params()['gbar_0'],
                        r_1=self._r_1, r_2=self._r_2, a_2=self._a_2)
        self.set_params({'r_2': 1.0, 'a_2': 0.0})

        z_bvp = np.linspace(self._z_min, self._z_max, bpts, endpoint=True)

        im_out = np.zeros_like(g)
        for m in np.ndindex(g.shape):
            self._gain.set_params({'gbar_0': g[m]})
            self.set_params({'r_1': r[m]})
            bvp = self.integrate_iz(z_bvp)
            ipm = bvp.sol(z_bvp)
            im_out[m] = ipm[1][0] * (1 - self._a_1 - r[m])

        self.set_params({
            'r_1': store.r_1,  # type: ignore
            'r_2': store.r_2,  # type: ignore
            'a_2': store.a_2  # type: ignore
        })
        self._gain.set_params({'gbar_0': store.gbar_0})  # type: ignore

        return im_out

    def plot_z(
        self,
        npts: int = 101,
        bpts: int = 15,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None
    ):
        '''
        Plot the forward and backward intensities as a function of the
        longitudinal coordinate z.

        Parameters
        ----------
        npts : int, optional
            Number of points to be used in the z array (default is 101)
        bpts : int, optional
            Initial number of nodes to be used to solve the boundary value
            problem (default is 15)
        show : bool, optional
            Flag to display the plots (default is True)
        savepath : str, optional
            Path to save the plots (default is None)
        filename : str, optional
            Name of the file to save the plots (default is None)
        '''
        z_bvp = np.linspace(0, self._z_max, bpts, endpoint=True)
        bvp = self.integrate_iz(z_bvp, verbose=2)

        z = np.linspace(self._z_min, self._z_max, npts, endpoint=True)
        ip_i = bvp.sol(z)
        ip_s = self.solve_iz(z)
        ip_x = self.approx_iz(z)

        gbar_0 = convert_to_float(self._gain.get_params()['gbar_0'])
        abar_0 = abs(convert_to_float(self._loss.get_params()['gbar_0']))
        annotation = (
            r'$\overline{{G}}_0 = {:.{prec}}$'.format(gbar_0, prec=3) + '\n'
            + (
                r'$\overline{{\alpha}}_0 = {:.{prec}}$'
                .format(abar_0, prec=2)
            ) + '\n'
            + r'$R_1 = {:.{prec}}$'.format(self._r_1, prec=3) + '\n'
            + r'$R_2 = {:.{prec}}$'.format(self._r_2, prec=3) + '\n'
            + self._gain.ls_annotation()
        )

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        ax.plot(  # type: ignore
            z, ip_i[0],
            label=r'$I^+(z)$' + ': integrate (bvp)'
        )
        ax.plot(  # type: ignore
            z, ip_s[0], '-.', label=r'$I^+(z)$' + ': solve (brentq)'
        )
        ax.plot(  # type: ignore
            z, ip_x[0], '--',
            label=r'$I^+(z)$' + ': approximate'
        )
        ax.plot(  # type: ignore
            z, ip_i[1],
            label=(
                r'$I^-(z)$'
                + ': integrate (bvp)'
            )
        )
        ax.plot(  # type: ignore
            z, ip_s[1], '-.', label=r'$I^-(z)$' + ': solve (brentq)'
        )
        ax.plot(  # type: ignore
            z, ip_x[1], '--',
            label=r'$I^-(z)$' + ': approximate'
        )
        ax.set_xlabel(r'$z$', fontdict=font)  # type: ignore
        ax.set_ylabel(r'$I^\pm(z)$', fontdict=font)  # type: ignore
        ax.set_xlim(z[0], z[-1])  # type: ignore
        ax.set_ylim(*get_ylim())
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        plt.legend(fontsize=labelsize, loc='upper left')  # type: ignore
        ob = offsetbox.AnchoredText(
            annotation,
            loc='lower right',
            pad=0,
            borderpad=0.65,
            prop=dict(size=fontsize)
        )
        ob.patch.set(  # type: ignore
            boxstyle='round',
            edgecolor='#D7D7D7',
            facecolor='white',
            alpha=0.75
        )
        ax.add_artist(ob)

        figdisp(fig, show, savepath, filename)

    def plot_sum(
        self,
        npts: int = 101,
        bpts: int = 15,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None
    ):
        '''
        Plot the sum of the forward and backward intensities as a function of
        the longitudinal coordinate z.

        Parameters
        ----------
        npts : int, optional
            Number of points to be used in the z array (default is 101)
        bpts : int, optional
            Initial number of nodes to be used to solve the boundary value
            problem (default is 15)
        show : bool, optional
            Flag to display the plots (default is True)
        savepath : str, optional
            Path to save the plots (default is None)
        filename : str, optional
            Name of the file to save the plots (default is None)
        '''
        z_bvp = np.linspace(self._z_min, self._z_max, bpts, endpoint=True)
        bvp = self.integrate_iz(z_bvp)

        z = np.linspace(self._z_min, self._z_max, npts, endpoint=True)
        ip_i = bvp.sol(z)
        ip_x = self.approx_iz(z)

        gbar_0 = convert_to_float(self._gain.get_params()['gbar_0'])
        abar_0 = abs(convert_to_float(self._loss.get_params()['gbar_0']))
        annotation = (
            r'$\overline{{G}}_0 = {:.{prec}}$'.format(gbar_0, prec=3) + '\n'
            + r'$\overline{{\alpha}}_0 = {:.{prec}}$'
              .format(abar_0, prec=2) + '\n'
            + r'$R_1 = {:.{prec}}$'.format(self._r_1, prec=3) + '\n'
            + r'$R_2 = {:.{prec}}$'.format(self._r_2, prec=3) + '\n'
            + self._gain.ls_annotation()
        )

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        ax.plot(z, ip_i[0] + ip_i[1], label='integrate (bvp)')  # type: ignore
        ax.plot(  # type: ignore
            z, ip_x[0] + ip_x[1], '--', label='approximate'
        )
        ax.plot(  # type: ignore
            z,
            np.mean(ip_i[0] + ip_i[1]) * np.ones_like(z),  # type: ignore
            '-.',
            label='average (bvp)'
        )
        ax.set_xlabel(r'$z$', fontdict=font)  # type: ignore
        ax.set_ylabel(r'$I^+(z) + I^-(z)$', fontdict=font)  # type: ignore
        ax.set_xlim(z[0], z[-1])
        ax.set_ylim(0, get_ylim()[1])
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        ax.legend(fontsize=labelsize, loc='lower left')  # type: ignore
        ob = offsetbox.AnchoredText(
            annotation,
            loc='lower right',
            pad=0,
            borderpad=0.65,
            prop=dict(size=fontsize)
        )
        ob.patch.set(  # type: ignore
            boxstyle='round',
            edgecolor='#D7D7D7',
            facecolor='white',
            alpha=0.75
        )
        ax.add_artist(ob)

        figdisp(fig, show, savepath, filename)


class LaserCWSHB(LaserCWSWL):
    '''
    Model a continuous-wave (CW) standing-wave laser (based on the Agrawal-Lax
    model) derived from LaserCWSWL
    '''

    def _kappa(self):
        '''
        Return the approximate effective value of kappa estimated using the
        Agrawal-Lax model.

        Returns
        -------
        numpy.float64
            Approximate value of kappa estimated using the Agrawal-Lax model
        '''
        r_1 = self._r_1
        r_2 = self._r_2
        rho = self._gain.rho()
        i_0 = self._intensity_0(self._config().kappa())
        phi_0 = (self._c2()/np.sqrt(r_1)) * i_0

        beta = (
            rho * (r_1 + r_2) * (1 + np.sqrt(r_1 * r_2))
            / (np.sqrt(r_1 * r_2) * (np.sqrt(r_1) + np.sqrt(r_2)))
        )
        phi = (3 / (2 * beta)) * (np.sqrt(1 + (8*beta/9) * phi_0) - 1)

        if phi_0 == phi/2:
            return 3
        else:
            return 2 * phi_0 / (phi_0 - phi/2)

    def _kappa_exact(self):
        '''
        Return the effective value of kappa computed using the Agrawal-Lax
        model.

        Returns
        -------
        numpy.float64
            Value of kappa computed using the Agrawal-Lax model
        '''
        i_rr = self._phi_rr() / np.sqrt(self._r_1)

        z = np.array([self._z_min, self._z_max])
        _, i_m = self.solve_iz(z)
        i_al = i_m[0]

        return 2 * i_rr / i_al

    def _di_dz(
        self,
        z: npt.NDArray[np.float64],
        iz: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Return the derivatives of the forward and backward intensities with
        respect to the longitudinal coordinate z.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z (normalized by the length of the laser
            cavity)
        iz : numpy.ndarray
            Forward and backward intensities at the longitudinal coordinates z

        Returns
        -------
        numpy.ndarray
            Derivatives of the forward and backward intensities with respect
            to the longitudinal coordinate z
        '''
        a = 1/self._gain.rho() + iz[0] + iz[1]
        b = 2 * np.sqrt(iz[0] * iz[1])

        dip_dz = (
            self._gain.gain(z)
            * ((iz[0] - a/2) / np.sqrt(a**2 - b**2) + 0.5)
            + self._loss.gain(z) * iz[0]
        )
        dim_dz = (
            -self._gain.gain(z)
            * ((iz[1] - a/2) / np.sqrt(a**2 - b**2) + 0.5)
            - self._loss.gain(z) * iz[1]
        )

        return np.vstack((dip_dz, dim_dz))

    def solve_iz(
        self,
        z: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        '''
        Return the exact solution for the forward and backward intensities at
        the longitudinal coordinates z.

        Parameters
        ----------
        z : numpy.ndarray
            Longitudinal coordinates z (normalized by the round-trip length of
            the laser cavity)
        '''
        def intensity_1(phi: float):
            r_1 = self._r_1
            rho = self._gain.rho()
            return (
                np.sqrt(4 * r_1 * phi**2 + (1 - r_1)**2 * rho**2 * phi**4)
                - (1 - r_1) * rho * phi**2
            ) / 2

        def intensity_2(phi: float) -> float:
            r_2 = self._r_2
            rho = self._gain.rho()
            return (
                np.sqrt(4 * r_2 * phi**2 + (1 - r_2)**2 * rho**2 * phi**4)
                + (1 - r_2) * rho * phi**2
            ) / (2 * r_2)

        def funcp(phi: float) -> float:
            rho = self._gain.rho()
            gbar_0 = convert_to_float(self._gain.get_params()['gbar_0'])
            return (
                np.log(intensity_2(phi) / intensity_1(phi)) / rho
                + (intensity_2(phi) - intensity_1(phi))
                * (
                    1
                    + phi**2
                    / (intensity_1(phi) * intensity_2(phi))
                )
                - gbar_0 / 2
            )

        def funcz(
            iz: float,
            z: float,
            i0: float,
            gbar_0: float,
            rho: float,
            phi: float
        ) -> float:
            return (
                np.log(iz / i0) / rho
                + (iz - i0) * (1 + phi**2 / (i0 * iz))
                - gbar_0 * z
            )

        gbar_0 = self._gain.get_params()['gbar_0']
        rho = self._gain.rho()

        phi_rr = self._phi_rr()
        phi = brentq(funcp, np.finfo(float).eps, np.sqrt(phi_rr))
        i_1 = intensity_1(phi)
        i_2 = intensity_2(phi)

        iz = np.zeros(len(z))
        iz[0] = i_1
        iz[-1] = i_2
        for n in range(len(z)-2):
            args = (z[n+1], i_1, gbar_0, rho, phi)
            iz[n+1] = brentq(funcz, i_1, i_2, args)

        ir = (iz + rho * phi**2)
        il = phi**2 * (rho + 1 / iz)

        return ir, il

    def plot_k(
        self,
        gbar_0: npt.NDArray[np.float64],
        npts: int = 101,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None
    ):
        '''
        Plot the value of kappa as a function of the reflectivity R given the
        integrated unsaturated round-trip gain gbar_0.

        Parameters
        ----------
        gbar_0 : numpy.ndarray
            Integrated unsaturated round-trip gain in the laser
        npts : int, optional
            Number of points to be used in the R array (default is 101)
        show : bool, optional
            Flag to display the plots (default is True)
        savepath : str, optional
            Path to save the plots (default is None)
        filename : str, optional
            Name of the file to save the plots (default is None)
        '''
        store = Storage(gbar_0=self._gain.get_params()['gbar_0'],
                        r_1=self._r_1, r_2=self._r_2, a_2=self._a_2)
        self.set_params({'r_2': 1.0, 'a_2': 0.0})

        r_mg, g_mg = self._grmg(gbar_0, npts)
        kappa = np.zeros_like(r_mg)
        kappa_x = np.zeros_like(r_mg)
        for m in np.ndindex(r_mg.shape):
            self._gain.set_params({'gbar_0': g_mg[m]})
            self.set_params({'r_1': r_mg[m]})
            kappa[m] = self._kappa_exact()
            kappa_x[m] = self._kappa()
        kappa[0, :] = 3.0
        kappa_x[0, :] = 3.0

        self.set_params({
            'r_1': store.r_1,  # type: ignore
            'r_2': store.r_2,  # type: ignore
            'a_2': store.a_2  # type: ignore
        })
        self._gain.set_params({'gbar_0': store.gbar_0})  # type: ignore

        annotation = (
            r'$\overline{{\alpha}}_0 = {:.{prec}}$'.format(
                abs(convert_to_float(self._loss.get_params()['gbar_0'])),
                prec=2
            )
            + '\n'
            + self._gain.ls_annotation()
        )

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        for n in range(gbar_0.size):
            ax.plot(r_mg[:, n], kappa[:, n],  # type: ignore
                    label=r'$\overline{{G}}_0 = {}$ (exact)'.format(gbar_0[n]))
            ax.plot(  # type: ignore
                r_mg[:, n], kappa_x[:, n], '--',
                label=(
                    r'$\overline{{G}}_0 = {}$ (approx)'
                    .format(gbar_0[n])
                )
            )
        ax.set_xlabel(r'$R$', fontdict=font)  # type: ignore
        ax.set_ylabel(r'$\kappa$', fontdict=font)  # type: ignore
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(2.0, 3.0)
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        ax.legend(fontsize=labelsize, loc='lower left')  # type: ignore
        ob = offsetbox.AnchoredText(
            annotation,
            loc='upper right',
            pad=0,
            borderpad=0.65,
            prop=dict(size=fontsize)
        )
        ob.patch.set(  # type: ignore
            boxstyle='round',
            edgecolor='#D7D7D7',
            facecolor='white',
            alpha=0.75
        )
        ax.add_artist(ob)

        figdisp(fig, show, savepath, filename)


class LaserAmplifierPulseProp(BaseClass):
    '''
    A virtual base-class wrapper for scipy.integrate.solve_ivp.

    Required Private Methods
    ------------------------
    _set_specs :
        Dict of keys that will be present in the parameter dictionary
        required by a derived class; default = dict()
    _it :
        Compute the current pulse intensity as a function of time;
        default: raises NotImplementedError
    _jt :
        Compute the current pulse fluence (the integrated pulse intensity)
        as a function of time; default: raises NotImplementedError

    Public Methods
    --------------
    integrate :
       Using scipy.integrate.solve_ivp, numerically integrate variables
       defined in _deriv; reports and plots results if successful
    '''

    _g_0: float
    _j_0: float
    _tau_pulse: float
    _omega: float

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.

        Parameters
        ----------
        params : dict
            A dict with the following keys (and values/types):
                'name' : str
                    A python string containing the name of the pulsed laser
                    model
                'g_0' : numpy.float64
                    Integrated single-pass unsaturated gain in the amplifier
                'j_0' : numpy.float64
                    Integrated input intensity (fluence, in units of
                    saturation fluence)
                'tau_pulse' : numpy.float64
                    Pulse width (in units of the end-to-end group propagation
                    time through the amplifier)
                'omega' : numpy.float64
                    Normalized dimensionless detuning frequency
        '''
        self._specs = {
            'name': {'units': ''},       # name of the object (text)
            'g_0': {'units': ''},        # integrated single-pass unsaturated
                                         # gain in the amplifier
            'j_0': {'units': ''},        # integrated input intensity (fluence,
                                         # in units of saturation fluence)
            'tau_pulse': {'units': ''},  # pulse width (in units of the
                                         # end-to-end group propagation time
                                         # through the amplifier)
            'omega': {'units': ''}       # normalized dimensionless detuning
                                         # frequency
        }

    def _it(
        self,
        t: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Compute the current input pulse intensity at time t.

        Parameters
        ----------
        t : float
            Time scaled by the group propagation time through the amplifier

        Returns
        -------
        retval : float
            The value of the input laser intensity time t
        '''
        raise NotImplementedError

    def _jt(
        self,
        t: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Compute the current pulse fluence (the integrated pulse intensity)
           at time t.
        Compute the current input pulse intensity at time t.

        Parameters
        ----------
        t : float
            Time scaled by the group propagation time through the amplifier

        Returns
        -------
        retval : float
            The value of the pulse fluence time t
        '''
        raise NotImplementedError

    def _gz(
        self,
        z: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Compute the gain as a function of position within the
        amplifier (0 < z < 1)

        Parameters
        ----------
        z : float
            Position scaled by the physical length of the amplifier

        Returns
        -------
        retval : float
            The value of the gain stored in the amplifier at position z
        '''
        return self._g_0 * (np.heaviside(z, 0.0) * np.heaviside(1 - z, 0.0))

    def _hz(
        self,
        z: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Compute the exponential integrated gain as a function of
        position within the amplifier (0 < z < 1)

        Parameters
        ----------
        z : float
            Position scaled by the physical length of the amplifier

        Returns
        -------
        retval : float
            The value of the exponential integrated gain stored in the
            amplifier at position z
        '''
        return np.exp(
            self._g_0 * (
                z * np.heaviside(z, 0.0) * np.heaviside(1 - z, 0.0)
                + np.heaviside(z - 1, 1.0)
            )
        )

    def intensity(
        self,
        z: float | npt.NDArray[np.float64],
        t: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Compute the laser intensity of a propagating pulse at position z
        and time t

        Parameters
        ----------
        z : float
            Position scaled by the physical length of the amplifier
        t : float
            Time scaled by the group propagation time through the amplifier

        Returns
        -------
        retval : float
            The value of the laser intensity at position z and time t
        '''
        it = self._it(t - z)
        jt = self._jt(t - z)
        hz = self._hz(z)

        return it / (1 + (1/hz - 1) * np.exp(-jt))

    def gain(
        self,
        z: float | npt.NDArray[np.float64],
        t: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Compute the gain stored within a laser amplifier at position z
        and time t

        Parameters
        ----------
        z : float
            Position scaled by the physical length of the amplifier
        t : float
            Time scaled by the group propagation time through the amplifier

        Returns
        -------
        retval : float
            The value of the laser amplifier gain at position z and time t
        '''
        jt = self._jt(t - z)
        gz = self._gz(z)
        hz = self._hz(z)

        return gz / (1 + (np.exp(jt) - 1) * hz)

    def effext(self) -> tuple[float, float]:
        '''
        Compute the effective net gain of a laser amplifier and the
        extraction efficiency of a laser pulse; both are independent
        of the input pulse shape and the dependence of the initial
        gain on position

        Parameters
        ----------
        z : float
            Position scaled by the physical length of the amplifier
        t : float
            Time scaled by the group propagation time through the amplifier

        Returns
        -------
        retval : tuple of floats
            The value of the effective net gain and the extraction efficiency
        '''
        h0 = np.exp(self._g_0 / (1 + self._omega**2))
        j1 = (
            (1 + self._omega**2)
            * np.log(
                1 + (np.exp(self._j_0 / (1 + self._omega**2)) - 1) * h0
            )
        )

        g_eff = j1 / self._j_0
        eta_ext = (j1 - self._j_0) / self._g_0

        return g_eff, eta_ext

    def plot(
        self,
        z: npt.NDArray[np.float64],
        t: float,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None
    ):
        '''
        Plot the laser intensity and amplifier gain over a range of positions
        at a particular time.

        Parameters
        ----------
        z : numpy.ndarry(dtype=numpy.float64)
            A vector of positions scaled by the physical length
            of the amplifier
        t : float
            A value of time scaled by the group propagation time through
            the amplifier
        show : Boolean
            Show a brief summary of the integration process and the plot of the
            integration variables over time
        savepath : string
            Path of the folder/directory where a copy of the figure will be
            saved using Matplotlib's savefig
        filename : string
            Name of the file (including the format extension) that will
            contain the figure; default = None; note that both savepath and
            filename must be specified so that Matplotlib's savefig target --
            savepath + filename -- is valid
        '''
        i = self.intensity(z, t)
        g = self.gain(z, t)

        g_eff, eta_ext = self.effext()
        annotation = (
            r'$\overline{{G}}_0 = {}$'.format(self._g_0) + '\n'
            + r'$J_0 = {}$'.format(self._j_0) + '\n'
            + r'$\Omega = {}$'.format(self._omega) + '\n'
            + r'$G_\mathrm{{eff}} = {:.{prec}}$'.format(g_eff, prec=4) + '\n'
            + r'$\eta_\mathrm{{ext}} = {:.{prec}}$'.format(eta_ext, prec=3)
        )

        if t < 0:
            loc_x = 0.705
        else:
            loc_x = 0.035
        loc_y = 0.59

        fig, ax1 = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        color = '#1f77b4'
        ax1.plot(z, i, color=color)  # type: ignore
        ax1.set_xlabel(r'$z$', fontdict=font)  # type: ignore
        ax1.set_ylabel(r'$I(z)$', fontdict=font, color=color)  # type: ignore
        ax1.tick_params(axis='x', labelsize=labelsize)  # type: ignore
        ax1.tick_params(  # type: ignore
            axis='y', labelsize=labelsize, labelcolor=color
        )
        ax1.set_xlim(z[0], z[-1])
        ax1.set_ylim(0.0, get_ylim()[1])
        ax1.text(  # type: ignore
            z[0] + loc_x * (z[-1] - z[0]),
            loc_y * get_ylim()[1],
            annotation,
            bbox=bbox,
            fontdict=font
        )
        ax2 = ax1.twinx()  # type: ignore
        color = '#ff7f0e'
        ax2.plot(z, g, color=color)  # type: ignore
        ax2.set_ylabel(r'$G(z)$', fontdict=font, color=color)  # type: ignore
        ax2.tick_params(  # type: ignore
            axis='y', labelsize=labelsize, labelcolor=color
        )
        ax2.set_ylim(0.0, get_ylim()[1])

        figdisp(fig, show, savepath, filename)

    def anim_pulse(
        self,
        z: npt.NDArray[np.float64],
        t: npt.NDArray[np.float64]
    ):
        '''
        Return an animation of the laser intensity and amplifier gain
        over a range of positions and times.

        Parameters
        ----------
        z : numpy.ndarry(dtype=numpy.float64)
            A vector of positions scaled by the physical length of the
            amplifier
        t : numpy.ndarry(dtype=numpy.float64)
            A vector of times scaled by the group propagation time through
            the amplifier

        Returns
        -------
        retval : tuple
            anim : matplotlib.animation.FuncAnimation
                The animation, which can be displayed simply by entering the
                variable name as a command
            i : numpy.ndarray(dtype=numpy.float64)
                A two-dimensional array containing I(z, t) values
                in the animation
            g : numpy.ndarray(dtype=numpy.float64)
                A two-dimensional array containing G(z, t) values
                in the animation
        '''
        z_mg, t_mg = np.meshgrid(z, t, sparse=False, indexing='ij')

        i = self.intensity(z_mg, t_mg)
        g = self.gain(z_mg, t_mg)
        g_eff, eta_ext = self.effext()
        print(
            (
                "Effective net pulse energy gain: {:.{prec}}"
                .format(g_eff, prec=4)
            )
        )
        print(
            "Energy extraction efficiency: {:.{prec}}".format(eta_ext, prec=3)
        )

        fig = plt.figure(figsize=(8, 4))  # type: ignore
        ax1 = plt.subplot(1, 1, 1)  # type: ignore

        txt_title = ax1.set_title('')  # type: ignore
        ax1.set_xlim(z[0], z[-1])
        ax1.set_ylim(0, get_ylim()[1])
        ax1.set_xlabel('z')  # type: ignore

        line1, = ax1.plot([], [], 'b', lw=2)  # type: ignore
        line2, = ax1.plot([], [], 'r', lw=2)  # type: ignore

        def drawframe(n: int):
            line1.set_data(z, g[:, n]/self._g_0)  # type: ignore
            line2.set_data(z, i[:, n]/self._j_0)  # type: ignore
            txt_title.set_text('Frame = {0:4d}'.format(n))  # type: ignore
            return (line1, line2)

        # blit=True re-draws only the parts that have changed.
        anim = animation.FuncAnimation(
            fig,
            drawframe,
            frames=t_mg.shape[1],
            interval=20,
            blit=True
        )

        return anim, i, g


class LaserAmplifierPulsePropRect(LaserAmplifierPulseProp):
    '''
    Model a rectangular input pulse propagating through a laser amplifier.
    '''
    def _it(
        self,
        t: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Compute the pulse intensity at time t for a rectangular
        input pulse.
        '''
        return (
            (self._j_0 / self._tau_pulse)
            * np.heaviside(t, 0.0)
            * np.heaviside(self._tau_pulse - t, 0.0)
        )

    def _jt(
        self,
        t: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Compute the pulse fluence (the integrated pulse intensity)
        at time t for a rectangular input pulse.
        '''
        return self._j_0 * (
            (t / self._tau_pulse)
            * np.heaviside(t, 0.0)
            * np.heaviside(self._tau_pulse - t, 0.0)
            + np.heaviside(t - self._tau_pulse, 1.0)
        )


class LaserAmplifierPulsePropGauss(LaserAmplifierPulseProp):
    '''
    Model a gaussian input pulse propagating through a laser amplifier.
    '''
    def _it(
        self,
        t: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Compute the pulse intensity at time t for a gaussian input pulse.
        '''
        return (
            self._j_0
            * (np.sqrt(2/np.pi) / self._tau_pulse)
            * np.exp(-2 * (t / self._tau_pulse) ** 2)
        )

    def _jt(
        self,
        t: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Compute the pulse fluence (the integrated pulse intensity)
        at time t for a gaussian input pulse.
        '''

        return (  # type: ignore
            (self._j_0 / 2)
            * (1 + erf(np.sqrt(2.0) * t / self._tau_pulse))
        )


class LaserIVPREA(ODEIntegrator):
    '''
    Model a laser (such as a Q-switched laser) using ODEIntegrator
    to solve the laser equations of motion for fluence and gain under the
    Rate Equation Approximation
    '''

    _tau_par: float
    _tau_pho: float
    _kappa: float

    def __init__(self, params: dict[str, object], pump: Shape, switch: Shape):
        '''
        Initialize a LaserPulsed object

        Parameters
        ----------
        params : dict
            A dict with the following keys (and values/types):
                'name' : str
                    A python string containing the name of the pulsed
                    laser model
                'tau_par': float
                    Gain lifetime in units of round-trip time
                'tau_pho': float
                    Photon lifetime in units of round-trip time
                'kappa' : int or float
                    URL: kappa = 1; SWL: kappa = 2
        pump : Shape
            Object providing the pump pulse shape function
        switch : Shape
            Object providing the switch shape function
        '''
        self._pump = pump
        self._switch = switch

        BaseClass.__init__(self, params)

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        '''
        self._specs = {
            'name': {'units': ''},               # name of the object
            'tau_par': {'units': '(tau_0)'},     # Gain lifetime
                                                 # (units of round-trip time)
            'tau_pho': {'units': '(tau_0)'},     # Photon lifetime
                                                 # (units of round-trip time)
            'kappa': {'units': ''}             # URL: kappa = 1; SWL: kappa = 2
        }

    def _p(
        self,
        t: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Return a pump pulse shape

        Parameters
        ----------
        t : float
            Time in units of laser cavity round-trip time

        Returns
        -------
        retval : float
            The value of the pump shape at time t
        '''
        return self._pump.shape(t)

    def _s(
        self,
        t: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Return a Q-switch pulse shape

        Parameters
        ----------
        t : float
            Time in units of laser cavity round-trip time

        Returns
        -------
        retval : float
            The value of the Q-switch shape at time t
        '''
        return self._switch.shape(t)

    def _se(self):
        '''
        Return a random number from NumPy's Gamma distribution as a fake
        representation of spontaneous emission noise; the proper model would
        use a Langevin noise simulation
        '''
        return 1.0e-06 * np.random.default_rng().gamma(2, 2)

    def _deriv(
        self,
        t: float,
        y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Derivatrive function of (t, y) that follows scipy.integrate.solve_ivp.
        Here y[0] is the intracavity fluence, and y[1] is the amplifier gain;
        required parameters are included in self._specs; required functions
        are self._p(t) (pump), self._s(t) (Q-switch), and self._se()
        (spontaneous emission noise)
        '''
        dydt = np.zeros_like(y)

        dydt[0] = (
            ((y[1] - 1) / self._tau_pho - self._s(t)) * y[0]
            + self._se()
        )
        dydt[1] = (
            (self._p(t) - y[1]) / self._tau_par
            - self._kappa * y[0] * y[1]
        )

        return dydt

    def _jac(
        self,
        t: float,
        y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''Return the Jacobian matrix of the derivative function _deriv'''
        return np.array([
            [
                (y[1] - 1) / self._tau_pho - self._s(t),
                y[0] / self._tau_pho
            ],
            [
                -self._kappa * y[1],
                -1.0 / self._tau_par - self._kappa * y[0]
            ]
        ])

    def _simplot(
        self,
        sol: Any,
        show: bool,
        savepath: str | None,
        filename: str | None
    ):
        '''
        Plot the intracavity fluence and gain as a function of time.

        Parameters
        ----------
        sol : bunch oblect
            Returned by scipy.integrate.solve_ivp
        show : Boolean
            Show a brief summary of the integration process and the plot of the
            integration variables over time
        savepath : string
            Path of the folder/directory where a copy of the figure will
            be saved
            using Matplotlib's savefig
        filename : string
            Name of the file (including the format extension) that will
            contain the figure; default = None; note that both savepath and
            filename must be specified so that Matplotlib's savefig target --
            savepath + filename -- is valid
        '''
        t = sol.t
        y = sol.y
        j = y[0]
        h = y[1]

        fig, ax1 = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        color = '#1f77b4'
        ax1.plot(t, j, color=color)  # type: ignore
        ax1.set_xlabel(r'$t$', fontdict=font)  # type: ignore
        ax1.set_ylabel(r'$J(t)$', fontdict=font, color=color)  # type: ignore
        ax1.tick_params(axis='x', labelsize=labelsize)  # type: ignore
        ax1.tick_params(  # type: ignore
            axis='y', labelsize=labelsize, labelcolor=color
        )
        ax1.set_xlim(t[0], t[-1])
        ax1.set_ylim(0.0, get_ylim()[1])
        ax2 = ax1.twinx()
        color = '#ff7f0e'
        ax2.plot(t, h, color=color)  # type: ignore
        ax2.set_ylabel(r'$H(t)$', fontdict=font, color=color)  # type: ignore
        ax2.tick_params(  # type: ignore
            axis='y', labelsize=labelsize, labelcolor=color
        )
        ax2.set_ylim(0.0, get_ylim()[1])

        figdisp(fig, show, savepath, filename)


class LaserIVPOMB(ODEIntegrator):
    '''
    Model a laser (e.g., gain-switched, modulated, or cw-pumped) as an
    initial-value problem using ODEIntegrator to solve the effective
    four-level Optical Maxwell-Bloch Equations; test the
    Rate Equation Approximation
    '''

    _tau_par: float
    _tau_prp: float
    _omega: float
    _tau_pho: float
    _kappa: float

    def __init__(self, params: dict[str, object], pump: Shape):
        '''
        Initialize a LaserIVPOMB object

        Parameters
        ----------
        params : dict
            A dict with the following keys (and values/types):
                'name' : str
                    A python string containing the name of the pulsed
                    laser model
                'tau_pho': float
                    Photon lifetime in units of round-trip time
                'tau_prp': float
                    Transverse decoherence time in units of round-trip time
                'tau_par': float
                    Gain lifetime in units of round-trip time
                'kappa' : int
                    URL: kappa = 1; SWL: kappa = 2
                'omega' : float
                    Normalized dimensionless detuning frequency
        pump : Shape
            Object providing the pump pulse shape function
        '''
        self._pump = pump

        BaseClass.__init__(self, params)

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        '''
        self._specs = {
            'name': {'units': ''},            # name of the object
            'tau_pho': {'units': '(tau_0)'},  # Photon lifetime
                                              # (units of round-trip time)
            'tau_prp': {'units': '(tau_0)'},  # Transverse decoherence time
                                              # (units of round-trip time)
            'tau_par': {'units': '(tau_0)'},  # Gain lifetime
                                              # (units of round-trip time)
            'kappa': {'units': ''},           # URL: kappa = 1; SWL: kappa = 2
            'omega': {'units': ''}            # Normalized dimensionless
                                              # detuning frequency
        }

    def _p(
        self,
        t: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        '''
        Return a pump pulse shape

        Parameters
        ----------
        t : float
            Time in units of laser cavity round-trip time

        Returns
        -------
        retval : float
            The value of the pump shape at time t
        '''
        return self._pump.shape(t)

    def _deriv(
        self,
        t: float,
        y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Derivative function of (t, y) that follows scipy.integrate.solve_ivp.
        The vector of state variables y has the components
            y[0]: The real part of the electric field
            y[1]: The imaginary part of electric field
            y[2]: The real part of the macroscopic polarization
            y[3]: The imaginary part of macroscopic polarization
            y[4]: The (real) amplifier gain
            y[5]: The real part of the electric field under the REA
            y[6]: The imaginary part of electric field under the REA
            y[7]: The (real) amplifier gain under the REA
        Required parameters are defined in self._specs; the only required
        function is self._p(t) (pump)
        '''
        dydt = np.zeros_like(y)

        dydt[0] = (-y[0] + self._omega * y[1] + y[2]) / (2 * self._tau_pho)
        dydt[1] = (-self._omega * y[0] - y[1] + y[3]) / (2 * self._tau_pho)
        dydt[2] = (
            -y[2]
            - self._omega * y[3]
            + (1 + self._omega**2) * y[0] * y[4]
        ) / self._tau_prp
        dydt[3] = (
            -y[3]
            + self._omega * y[2]
            + (1 + self._omega**2) * y[1] * y[4]
        ) / self._tau_prp
        dydt[4] = (
            self._p(t)
            - y[4]
            - self._kappa * (y[0] * y[2] + y[1] * y[3]) / (1 + self._omega**2)
        ) / self._tau_par

        dydt[5] = (
            (y[5] - self._omega * y[6])
            * (y[7] - 1)
            / (2 * self._tau_pho)
        )
        dydt[6] = (
            (self._omega * y[5] + y[6]) * (y[7] - 1)
            / (2 * self._tau_pho)
        )
        dydt[7] = (
            self._p(t)
            - y[7] * (
                1
                + self._kappa * (y[5]**2 + y[6]**2) / (1 + self._omega**2)
            )
        ) / self._tau_par

        return dydt

    def _jac(
        self,
        t: float,
        y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Jacobian function of (t, y) that follows scipy.integrate.solve_ivp.
        The vector of state variables y has the components
            y[0]: The real part of the electric field
            y[1]: The imaginary part of electric field
            y[2]: The real part of the macroscopic polarization
            y[3]: The imaginary part of macroscopic polarization
            y[4]: The (real) amplifier gain
            y[5]: The real part of the electric field under the REA
            y[6]: The imaginary part of electric field under the REA
            y[7]: The (real) amplifier gain under the REA
        Required parameters are defined in self._specs.
        '''
        jac = np.zeros((8, 8))

        one_plus_omega_squared = 1 + self._omega**2
        factor_pho = 1 / (2 * self._tau_pho)
        factor_prp = one_plus_omega_squared / self._tau_prp
        factor_par = self._kappa / (one_plus_omega_squared * self._tau_par)

        jac[0, 0] = -factor_pho
        jac[0, 1] = factor_pho * self._omega
        jac[0, 2] = factor_pho

        jac[1, 0] = -factor_pho * self._omega
        jac[1, 1] = -factor_pho
        jac[1, 3] = factor_pho

        jac[2, 0] = factor_prp * y[4]
        jac[2, 2] = -1 / self._tau_prp
        jac[2, 3] = -self._omega / self._tau_prp
        jac[2, 4] = factor_prp * y[0]

        jac[3, 1] = factor_prp * y[4]
        jac[3, 2] = self._omega / self._tau_prp
        jac[3, 3] = -1 / self._tau_prp
        jac[3, 4] = factor_prp * y[1]

        jac[4, 0] = -factor_par * y[2]
        jac[4, 1] = -factor_par * y[3]
        jac[4, 2] = -factor_par * y[0]
        jac[4, 3] = -factor_par * y[1]
        jac[4, 4] = -1 / self._tau_par

        jac[5, 5] = factor_pho * (y[7] - 1)
        jac[5, 6] = -factor_pho * self._omega * (y[7] - 1)
        jac[5, 7] = factor_pho * (y[5] - self._omega * y[6])

        jac[6, 5] = factor_pho * self._omega * (y[7] - 1)
        jac[6, 6] = factor_pho * (y[7] - 1)
        jac[6, 7] = factor_pho * (self._omega * y[5] + y[6])

        jac[7, 5] = -2 * factor_par * y[5] * y[7]
        jac[7, 6] = -2 * factor_par * y[6] * y[7]
        jac[7, 7] = -1 / self._tau_par - factor_par * (y[5]**2 + y[6]**2)

        return jac

    def _simplot(
        self,
        sol: Any,
        show: bool,
        savepath: str | None,
        filename: str | None
    ):
        '''
        Plot the intracavity intensities, gain, and polarization phase
        as a function of time for both the OMB ODEs and the REA ODEs.

        Parameters
        ----------
        sol : bunch oblect
            Returned by scipy.integrate.solve_ivp
        show : Boolean
            Show a brief summary of the integration process and the plot of the
            integration variables over time
        savepath : string
            Path of the folder/directory where a copy of the figure will be
            saved using Matplotlib's savefig
        filename : string
            Base name of the files (including the format extension) that will
            contain the figures; default = None; note that both savepath
            and filename must be specified so that Matplotlib's savefig target
            -- savepath + filename -- is valid.

        Note
        ----
        The filename is modified by appending '_e', '_h', and '_p',
        so that the three plots are saved seaparately as
        filename + '_e', filename + '_h', and filename + '_p'.
        '''
        filenames = [
            modname(filename, '_e'),
            modname(filename, '_h'),
            modname(filename, '_p')
        ]

        self._plot_e(sol, show, savepath, filenames[0])
        self._plot_h(sol, show, savepath, filenames[1])
        self._plot_p(sol, show, savepath, filenames[2])

    def _plot_e(
        self,
        sol: Any,
        show: bool,
        savepath: str | None,
        filename: str | None
    ):
        '''
        Plot the intracavity intensities as a function of time
        for both the OMB ODEs and the REA ODEs. Input parameters
        are identical to those of self._simplot().
        '''
        t = sol.t
        y = sol.y
        e = y[0] + 1j * y[1]
        e_rea = y[5] + 1j * y[6]

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        ax.plot(t, np.abs(e)**2, label='OMB')  # type: ignore
        ax.plot(t, np.abs(e_rea)**2, '--', label='REA')  # type: ignore
        ax.set_xlabel(r'$t$', fontdict=font)  # type: ignore
        ax.set_ylabel(r'$|E(t)|^2$', fontdict=font)  # type: ignore
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0.0, get_ylim()[1])
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.legend(loc='upper right', fontsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore

        figdisp(fig, show, savepath, filename)

    def _plot_h(
        self,
        sol: Any,
        show: bool,
        savepath: str | None,
        filename: str | None
    ):

        '''
        Plot the intracavity gain as a function of time for both the
        OMB ODEs and the REA ODEs. Input parameters are identical
        to those of self._simplot().
        '''
        t = sol.t
        y = sol.y
        h = y[4]
        h_rea = y[7]

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        ax.plot(t, h, label='OMB')  # type: ignore
        ax.plot(t, h_rea, '--', label='REA')  # type: ignore
        ax.set_xlabel(r'$t$', fontdict=font)  # type: ignore
        ax.set_ylabel(r'$H(t)$', fontdict=font)  # type: ignore
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(*get_ylim())
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.legend(loc='upper right', fontsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore

        figdisp(fig, show, savepath, filename)

    def _plot_p(
        self,
        sol: Any,
        show: bool,
        savepath: str | None,
        filename: str | None
    ):
        '''
        Plot the intracavity polarization phase as a function of time
        for both the OMB ODEs and the REA ODEs. Input parameters are
        identical to those of self._simplot().
        '''
        t = sol.t
        y = sol.y
        e = y[0] + 1j * y[1]
        f = y[2] + 1j * y[3]
        e_rea = y[5] + 1j * y[6]
        h_rea = y[7]
        f_rea = (1 + 1j * self._omega) * h_rea * e_rea

        dphi = np.array(np.angle(f * np.conj(e)), dtype=np.float64)
        dphi_rea = np.array(np.angle(f_rea * np.conj(e_rea)), dtype=np.float64)

        idx = np.abs(dphi - np.sign(dphi) * np.pi) < 1.0e-02 * np.pi
        dphi[idx] = dphi[idx] + np.sign(-dphi[idx]) * np.pi

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        ax.plot(t, dphi, label='OMB')  # type: ignore
        ax.plot(t, dphi_rea, '--', label='REA')  # type: ignore
        ax.set_xlabel(r'$t$', fontdict=font)  # type: ignore
        ax.set_ylabel(  # type: ignore
            r'$\phi_F(t) - \phi_E(t)$', fontdict=font
        )
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(dphi_rea[-1] - 0.1, dphi_rea[-1] + 0.1)
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.legend(loc='upper right', fontsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore

        figdisp(fig, show, savepath, filename)
