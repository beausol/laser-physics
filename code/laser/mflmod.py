import numpy as np
import numpy.typing as npt
from scipy.special import erf

import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

from laser.utils import BaseClass, Storage, get_ylim, figdisp, manexp10
from laser.utils import font, labelsize


class ExponentialSignal(BaseClass):
    """
    Represent signals having the form exp( f(t) ), where
        f(t) = (-gamma + i * lambda) * t - 0.5 * (a + i * b) * t^2

    Required Private Methods
    ------------------------
    _set_specs :
        Dict of keys that will be present in the parameter dictionary
        required by a derived class; default = dict()
    _f :
        Compute the function
            f(t) = (-gamma + i * lambda) * t
                 - 0.5 * (a + i * b) * t^2
        where
            gamma, lambda, a, and b are parameters of the signal.
    _e_q :
        Compute the Fourier coefficients of the field.
    _annotate :
        Annotate the plots with the values of the parameters listed in
        _set_specs

    Public Methods
    --------------
    get_eq :
        Get the Fourier coefficients of the field
    plot_e2 :
        Plot the magnitude squared of the field at times -0.5 <= t <= 0.5
    plot_phidot :
        Plot the phase of the field at times -0.5 <= t <= 0.5
    plot_eq2 :
        Plot the magnitude squared of the Fourier coefficients of the field
    plot_phiq :
        Plot the phase of the Fourier coefficients of the field
    """

    _gamma: float
    _lamda: float
    _a: float
    _b: float

    def _set_specs(self):
        """
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        """
        self._specs = {
            "name": {"units": ""},  # name of the object (text)
            "gamma": {"units": ""},  # real linear parameter
            "lamda": {"units": ""},  # imaginary linear parameter
            "a": {"units": ""},  # real quadratic parameter
            "b": {"units": ""}  # imaginary quadratic parameter
        }

    def _f(self, t: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """
        Compute the function f(t) = (-gamma + i * lambda) * t
        - 0.5 * (a + i * b) * t^2.

        Parameters
        ----------
        t : np.ndarray
            Time values at which to evaluate the function
        gamma : float
            Real linear parameter

        Returns
        -------
        np.ndarray
            Computed values of f(t)
        """
        return (
            (-self._gamma + 1j * self._lamda) * t
            - 0.5 * (self._a + 1j * self._b) * t**2
        )

    def _e_q(
        self, q: int | npt.NDArray[np.int32]
    ) -> npt.NDArray[np.complex128]:
        """
        Compute the Fourier coefficients of the field.

        Parameters
        ----------
        q : int | numpy.ndarray
            The index of the Fourier coefficient to compute; may be a scalar
            or an array

        Returns
        -------
        numpy.ndarray
            Array of Fourier coefficients
        """
        gl = self._gamma - 1j * (self._lamda - 2 * q * np.pi)
        ab = self._a + 1j * self._b

        sqrt_term = np.sqrt(np.pi / (2 * ab))
        exp_term = np.exp(gl**2 / (2 * ab))
        erf_arg1 = (0.5 * ab + gl) / np.sqrt(2 * ab)
        erf_arg2 = (0.5 * ab - gl) / np.sqrt(2 * ab)
        erf_sum = erf(erf_arg1) + erf(erf_arg2)  # type: ignore

        return sqrt_term * exp_term * erf_sum  # type: ignore

    def _annotate(self, ax: plt.Axes, loc: str):
        """
        Annotate the plot with the values of the parameters.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to annotate the plot
        loc : str
            The location of the annotation; one of 'upper left', 'upper right',
            'lower left', 'lower right', 'upper center', 'lower center',
            'center left', 'center right', 'center', or 'best'
        """
        template = "$\\gamma = {}$; $\\lambda_0 = {}$\n$a = {}$; $b = {}$"
        annotation = template.format(
            self._gamma, self._lamda, self._a, int(self._b)
        )
        ob = offsetbox.AnchoredText(
            annotation,
            loc=loc,
            pad=0,
            borderpad=0.65,
            prop=dict(size=labelsize),
        )
        ob.patch.set(  # type: ignore
            boxstyle="round",
            edgecolor="#D7D7D7",
            facecolor="white",
            alpha=0.75,
        )
        ax.add_artist(ob)

    def _field(self, t: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """
        Compute the field at time t.
        """
        return np.exp(self._f(t))

    def get_eq(self, q_max: int) -> npt.NDArray[np.complex128]:
        """
        Get the Fourier coefficients of the field.

        Parameters
        ----------
        q_max : int
            Let q = -q_max, -q_max + 1, ..., 0, ..., q_max - 1, q_max

        Returns
        -------
        numpy.ndarray
            Array of Fourier coefficients for the specified values of q
        """
        q = np.arange(-q_max, q_max + 1)
        return self._e_q(q)

    def fwhm(self, q_max: int) -> int:
        """
        Determine the full-width at half-maximum of the squared Fourier]
        coefficients of the field.

        Parameters
        ----------
        q_max : int
            Maximum value of |q| to consider

        Returns
        -------
        float
            Full-width at half-maximum of the squared Fourier coefficients
            (i.e., the number of modes with squared magnitude greater than or
            equal to 0.5 times the maximum squared magnitude)
        """
        q = np.arange(-q_max, q_max + 1)
        e2_q = np.abs(self._e_q(q)) ** 2
        e2_max = np.max(e2_q)
        modes = np.where(e2_q >= 0.5 * e2_max)[0]

        return max(modes) - min(modes)

    def fwhx(self):
        """
        Estimate the full-width at half-maximum of the squared Fourier
        coefficients of the field.

        Returns
        -------
        float
            Estimated full-width at half-maximum of the squared Fourier
            coefficients (i.e., the number of modes with squared magnitude
            greater than or equal to 0.5 times the maximum squared magnitude)
        """
        a = self._a
        b = self._b
        c = 4 * np.log(2)
        numerator = (np.log(2) / np.pi**2) * (a**2 + b**2)
        denominator = np.sqrt(a**2 + c**2)
        modes = int(round(np.sqrt(float(numerator / denominator))))

        return modes

    def plot_e2(self, num: int = 1001):
        """
        Plot the magnitude squared of the field at times -0.5 <= t <= 0.5.

        Parameters
        ----------
        num : int
            Number of points to plot; default = 1001
        """
        t = np.linspace(-0.5, 0.5, num=num, endpoint=True)

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        ax.plot(t, np.abs(self._field(t))**2)  # type: ignore
        ax.set_xlabel(r"$t$", fontdict=font)  # type: ignore
        ax.set_ylabel(r"$|E(t)|^2$", fontdict=font)  # type: ignore
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0.0, get_ylim()[1])
        ax.tick_params(axis="both", labelsize=labelsize)  # type: ignore
        ax.set_xticks(np.linspace(-0.5, 0.5, num=11))
        ax.grid(visible=True)  # type: ignore
        self._annotate(ax, loc="lower right")

        plt.show()  # type: ignore

    def plot_phidot(self, num: int = 1001):
        """
        Plot the phase of the field at times -0.5 <= t <= 0.5.

        Parameters
        ----------
        num : int
            Number of points to plot; default = 1001
        """
        t = np.linspace(-0.5, 0.5, num=num, endpoint=True)

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        phase = np.angle(self._field(t))
        unwrapped_phase = np.unwrap(phase)
        phase_derivative = np.gradient(unwrapped_phase, t)
        ax.plot(t, phase_derivative)  # type: ignore
        ax.set_xlabel(r"$t$", fontdict=font)  # type: ignore
        ax.set_ylabel(r"$\dot{\phi(t)}$", fontdict=font)  # type: ignore
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(*get_ylim())
        ax.tick_params(axis="both", labelsize=labelsize)  # type: ignore
        ax.set_xticks(np.linspace(-0.5, 0.5, num=11))
        ax.grid(visible=True)  # type: ignore
        self._annotate(ax, loc="lower right")

        plt.show()  # type: ignore

    def plot_eq2(
        self,
        q_max: int,
        db: bool = True,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None,
    ):
        """
        Plot the magnitude squared of the Fourier coefficients of the field.

        Parameters
        ----------
        q_max : int
            Maximum value of |q| to plot
        db : bool
            If True, plot in decibels; default = True
        show : bool
            If True, display the plot; default = True
        savepath : str | None
            Path of the folder/directory where a copy of the figure will be
            saved using Matplotlib's savefig; default = None
        filename : str | None
            Name of the file (including the format extension) that will
            contain the figure; default = None; note that both savepath
            and filename must be specified so that Matplotlib's savefig
            target -- savepath + filename -- is valid
        """
        q = np.arange(-q_max, q_max + 1)
        e_q = self._e_q(q)

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        if db:
            ax.stem(  # type: ignore
                q,
                10 * (
                    np.log10(np.abs(e_q) ** 2)
                    - np.min(np.log10(np.abs(e_q) ** 2))
                ),
                "-",
            )
            ax.set_ylabel(r"$|E_q|^2$~(dB)", fontdict=font)  # type: ignore
        else:
            ax.stem(q, np.abs(e_q) ** 2, "-")  # type: ignore
            ax.set_ylabel(r"$|E_q|^2$", fontdict=font)  # type: ignore
        ax.set_xlabel(r"$q$", fontdict=font)  # type: ignore
        ax.set_xlim(q[0], q[-1])
        ax.set_ylim(0.0, get_ylim()[1])
        ax.tick_params(axis="both", labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        if self._b < 0:
            loc = "upper right"
        else:
            loc = "upper left"
        self._annotate(ax, loc=loc)

        figdisp(fig, show, savepath, filename)

    def plot_phiq(self, q_max: int, show: bool = True,
                  savepath: str | None = None, filename: str | None = None):
        """
        Plot the phase of the Fourier coefficients of the field.

        Parameters
        ----------
        q_max : int
            Maximum value of |q| to plot
        show : bool
            If True, display the plot; default = True
        savepath : str | None
            Path of the folder/directory where a copy of the figure will be
            saved using Matplotlib's savefig; default = None
        filename : str | None
            Name of the file (including the format extension) that will
            contain the figure; default = None; note that both savepath
            and filename must be specified so that Matplotlib's savefig
            target -- savepath + filename -- is valid

        """
        q = np.arange(-q_max, q_max + 1)
        phi_q = np.unwrap(np.angle(self._e_q(q)))
        phi_0 = phi_q[phi_q.size // 2]
        phi_x = (
            phi_0
            + 2 * np.pi**2 * q**2 * self._b / (self._a**2 + self._b**2)
        )

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        ax.stem(q, phi_q, label="Exact")  # type: ignore
        ax.plot(q, phi_x, color="C1", label="Approx")  # type: ignore
        ax.set_xlabel(r"$q$", fontdict=font)  # type: ignore
        ax.set_ylabel(r"$\phi_q$", fontdict=font)  # type: ignore
        ax.set_xlim(q[0], q[-1])
        ax.set_ylim(*get_ylim())
        ax.tick_params(axis="both", labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        handles, labels = ax.get_legend_handles_labels()  # type: ignore
        if self._b < 0:
            loc = "upper right"
        else:
            loc = "lower right"
        order = [1, 0]
        ax.legend(  # type: ignore
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc=loc,
            fontsize=labelsize,
        )
        if self._b < 0:
            loc = "upper left"
        else:
            loc = "lower left"
        self._annotate(ax, loc=loc)

        figdisp(fig, show, savepath, filename)


class MFLaserModelQuad(BaseClass):
    """
    Calcualate and plot the parameters of an analytic Mean-Field Laser model
    with quadratic amplitude and phase profiles.
    """

    _r_1: float
    _r_2: float
    _tau_pho: float
    _tau_par: float
    _tau_prp: float
    _d_2: float
    _lef: float

    def _set_specs(self):
        """
        Required by BaseClass. Specify the names (strings) of dictionary keys
        that must be supplied through __init__() and from them create a private
        dictionary _specs.
        """
        self._specs = {
            "name": {"units": ""},  # name of the object (text)
            "r_1": {"units": ""},  # reflectivity of mirror 1
            "r_2": {"units": ""},  # reflectivity of mirror 2
            "tau_pho": {
                "units": "(tau_0)"
            },  # Photon lifetime (units of round-trip time)
            "tau_par": {"units": "(tau_0)"},  # Gain lifetime
            # (units of round-trip time)
            "tau_prp": {
                "units": "(tau_0)"
            },  # Transverse decoherence time (units of round-trip time)
            "d_2": {"units": ""},  # Normalized dispersion parameter
            "lef": {"units": ""},  # Linewidth enhancement factor
        }

    def _check_h0(
        self,
        h_0: float | npt.NDArray[np.float64],
    ) -> float | npt.NDArray[np.float64]:
        """Check the relative gain h_0.

        Parameters
        ----------
        h_0 : float | numpy.ndarray
            Relative gain value(s); must be greater than or equal to 1
            (above threshold). If  h_0 - 1 is less than or equal to 10 times
            the machine epsilon, h_0 is incremented by 1.0e-06 to avoid
            numerical issues in the computations and plots.

        Raises
        ------
        ValueError
            If h_0 < 1
        """
        if isinstance(h_0, float):
            if h_0 < 1:
                raise ValueError("h_0 must be greater than or equal to 1")
            if h_0 - 1 <= 10 * np.finfo(float).eps:
                h_0 += 1.0e-06
        elif isinstance(h_0, np.ndarray):
            if np.any(h_0 < 1):
                raise ValueError("h_0 must be greater than or equal to 1")
            if np.any(h_0 - 1 <= 10 * np.finfo(float).eps):
                h_0[h_0 - 1 <= 10 * np.finfo(float).eps] += 1.0e-06

        return h_0

    def _check_lef(self):
        """
        Check that the absolute value of the linewidth enhancement factor
        is zero.

        Raises
        ------
        ValueError
            If abs(self._lef) is larger than 10 times the machine epsilon
        """
        if np.abs(self._lef) > 10 * np.finfo(float).eps:
            raise ValueError(
                "LEF must be zero for the current implementation of this \
                    function"
            )

    def mu(self) -> float:
        """
        Compute the effective phase relaxation parameter mu.

        Returns
        -------
        float
            Effective phase relaxation parameter value(s)
        """

        r_1 = self._r_1
        r_2 = self._r_2
        tau_par = self._tau_par
        tau_prp = self._tau_prp
        kappa = 2.0  # Assuming kappa is always 2 for this implementation

        mu = (
            (2 * tau_par + 3 * tau_prp) / (4 * kappa)
        ) * np.log(1 / np.sqrt(r_1 * r_2))

        return mu

    def amp02(
        self,
        h_0: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """
        Compute the amplitude squared of the unperturned field envelope
        function A_0^2 as a function of the relative gain H_0
        (assuming that a = 0)

        Parameters
        ----------
        h_0 : float | numpy.ndarray
            Relative gain value(s)

        Returns
        -------
        float | numpy.ndarray
            Unperturbed amplitude squared value(s)

        Notes
        -----
        This function will not give correct results if h_0 < 1, and it is
        recommended that h_0 - 1 > 1.0e-06.
        """
        h_0 = self._check_h0(h_0)

        mu = self.mu()
        sigma = (h_0 - 1) / h_0

        amp02 = (
            1
            - ((2 + mu) - np.sqrt((2 + mu) ** 2 - 8 * mu * sigma))
            / (4 * sigma)
        )

        return amp02

    def rcal(
        self,
        h_0: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """
        Compute the static saturation parameter R(A) as a function of the
        relative gain H_0.

        Parameters
        ----------
        h_0 : float | numpy.ndarray
            Relative gain value(s)

        Returns
        -------
        float | numpy.ndarray
            Static saturation parameter value(s)

        Notes
        -----
        This function will not give correct results if h_0 < 1, and it is
        recommended that h_0 - 1 > 1.0e-06.
        """
        h_0 = self._check_h0(h_0)

        sigma = (h_0 - 1) / h_0
        amp = np.sqrt(self.amp02(h_0))

        r = 1 / (1 + sigma * (amp - 1))

        return r

    def amp2(
        self,
        h_0: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """
        Compute the amplitude squared of the field envelope function A^2 as a
        function of the relative gain H_0 (assumeing that LEF = 0).

        Parameters
        ----------
        h_0 : float | numpy.ndarray
            Relative gain value(s)

        Returns
        -------
        float | numpy.ndarray
            Amplitude squared value(s)

        Notes
        -----
        This function will not give correct results if h_0 < 1, and it is
        recommended that h_0 - 1 > 1.0e-06.
        """
        h_0 = self._check_h0(h_0)
        self._check_lef()

        mu = self.mu()
        sigma = (h_0 - 1) / h_0

        amp2 = self.amp02(h_0) - (2 - sigma) * mu * self.a(h_0) / 3

        return amp2

    def gamma(self) -> float:
        """
        Estimate the real linear parameter gamma for a quadratic exponential
        signal.

        Returns
        -------
        float
            Real linear parameter value
        """
        return 1 / self._tau_pho

    def lamda(
        self,
        h_0: float | npt.NDArray[np.float64],
    ) -> float | npt.NDArray[np.float64]:
        """
        Compute the linear phase shift parameter lambda as a function of the
        relative gain H_0 (assuming that LEF = 0).

        Parameters
        ----------
        h_0 : float | numpy.ndarray
            Relative gain value(s)

        Returns
        -------
        float | numpy.ndarray
            Linear phase shift parameter value(s)

        Notes
        -----
        This function will not give correct results if h_0 < 1, and it is
        recommended that h_0 - 1 > 1.0e-06.
        """
        h_0 = self._check_h0(h_0)
        self._check_lef()

        tau_pho = self._tau_pho
        d2 = self._d_2

        amp02 = self.amp02(h_0)
        r = self.rcal(h_0)
        mu = self.mu()
        sigma = (h_0 - 1) / h_0

        lamda = amp02**2 * r**4 * (mu**2 / 24) * sigma**2 / (d2 * tau_pho**2)

        return lamda

    def a(
        self,
        h_0: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """
        Compute the real quadratic parameter a as a function of the relative
        gain H_0 (allowing nonzero values of LEF).

        Parameters
        ----------
        h_0 : float | numpy.ndarray
            Relative gain value(s)

        Returns
        -------
        float | numpy.ndarray
            Real quadratic parameter value(s)

        Notes
        -----
        This function will not give correct results if h_0 < 1, and it is
        recommended that h_0 - 1 > 1.0e-06.
        """
        h_0 = self._check_h0(h_0)

        tau_pho = self._tau_pho
        tau_prp = self._tau_prp
        lef = self._lef
        d2 = self._d_2 + (4 * tau_prp**2 / (2 * tau_pho + tau_prp)) * (
            lef / (1 + lef**2)
        )

        amp02 = self.amp02(h_0)
        r = self.rcal(h_0)
        mu = self.mu()
        sigma = (h_0 - 1) / h_0

        a = (
            amp02
            * r**4
            * (mu**2 * sigma / (2 * (2 + 5 * mu)))
            * (1 / (d2 * tau_pho)) ** 2
            * (tau_prp**2 / ((1 + lef**2) * h_0) + 2 * lef * d2 * tau_pho)
        )

        return a

    def b(
        self,
        h_0: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """
        Compute the imaginary quadratic parameter b as a function of the
        relative gain H_0 (assuming that LEF = 0).

        Parameters
        ----------
        h_0 : float | numpy.ndarray
            Relative gain value(s)

        Returns
        -------
        float | numpy.ndarray
            Imaginary quadratic parameter value(s)

        Notes
        -----
        This function will not give correct results if h_0 < 1, and it is
        recommended that h_0 - 1 > 1.0e-06.
        """
        h_0 = self._check_h0(h_0)
        self._check_lef()

        tau_pho = self._tau_pho
        d2 = self._d_2

        mu = self.mu()
        sigma = (h_0 - 1) / h_0
        amp02 = self.amp02(h_0)
        r = self.rcal(h_0)

        b = amp02 * r**2 * (mu * sigma) / (2 * d2 * tau_pho) - mu * sigma * (
            2 * sigma - 1
        ) * self.a(h_0) / (6 * d2 * tau_pho)

        return b

    def get_expsig_params(
        self,
        h_0: float | npt.NDArray[np.float64]
    ) -> dict[str, float]:
        """
        Get the values of the exponential signal parameters of the model at
        the current state for a particular value of the relative gain h_0.

        Parameters
        ----------
        h_0 : float | numpy.ndarray
            Relative gain value; must be greater than or equal to 1
            (above threshold). If h_0 - 1 is less than or equal to 10 times
            the machine epsilon, h_0 is incremented by 1.0e-06 to avoid
            numerical issues in the computations and plots. If h_0 is a
            numpy.ndarray, it is assumed that h_0[0] is the value to use.

        Raises
        ------
        ValueError
            If h_0 < 1
        TypeError
            If h_0 is not a float or numpy.ndarray

        Returns
        -------
        dict
            Dictionary of parameter values (gamma, lamda, a, b)
        """
        self._check_h0(h_0)
        if not isinstance(h_0, float):
            if isinstance(h_0, np.ndarray):
                h_0 = h_0[0]
            raise TypeError("h_0 must be a float or numpy.ndarray")
        gamma = float(self.gamma())
        lamda = float(self.lamda(h_0))
        a = float(self.a(h_0))
        b = float(self.b(h_0))

        return {"gamma": gamma, "lamda": lamda, "a": a, "b": b}

    def plot_amp2(
        self,
        h_0: npt.NDArray[np.float64],
        d_2_arr: npt.NDArray[np.float64] | None = None,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None,
    ):
        """
        Plot the amplitude squared as a function of the relative gain H_0.

        Parameters
        ----------
        h_0 : numpy.ndarray
            Array of relative gain values
        d_2_arr : numpy.ndarray
            Array of normalized dispersion parameter values;
            default = None (use the value from the object)
        show : bool
            If True, display the plot; default = True
        savepath : str
            Path to save the plot; default = None
        filename : str
            Name of the file to save the plot; default = None
        """
        self._check_h0(h_0)
        store = Storage(d_2=self._d_2)

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        if d_2_arr is not None:
            for d_2 in d_2_arr:
                self.set_params({"d_2": d_2})
                man, exp = manexp10(d_2)
                ax.plot(  # type: ignore
                    h_0,
                    self.amp2(h_0) - 1,
                    label="$D_2 = {} \\times 10^{{{}}}$".format(man, exp),
                )
            legend = True
            loc = "upper right"
        else:
            ax.plot(h_0, self.amp2(h_0) - 1)  # type: ignore
            legend = False
        ax.set_xlabel(r"$\overline{H}_0$", fontdict=font)  # type: ignore
        ax.set_ylabel(r"$|A|^2 - 1$", fontdict=font)  # type: ignore
        ax.set_xlim(h_0[0], h_0[-1])
        ax.set_ylim(*get_ylim())
        ax.set_xticks(np.arange(h_0[0], h_0[-1] + 1))
        ax.tick_params(axis="both", labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        if legend:
            ax.legend(loc=loc, fontsize=labelsize)  # type: ignore

        self.set_params({"d_2": store.d_2})  # type: ignore

        figdisp(fig, show, savepath, filename)

    def plot_lamda(
        self,
        h_0: npt.NDArray[np.float64],
        d_2_arr: npt.NDArray[np.float64] | None = None,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None,
    ):
        """
        Plot the phase relaxation parameter lambda as a function of the
        relative gain H_0.

        Parameters
        ----------
        h_0 : numpy.ndarray
            Array of relative gain values
        d_2_arr : numpy.ndarray
            Array of normalized dispersion parameter values;
            default = None (use the value from the object)
        show : bool
            If True, display the plot; default = True
        savepath : str
            Path to save the plot; default = None
        filename : str
            Name of the file to save the plot; default = None
        """
        self._check_h0(h_0)
        store = Storage(d_2=self._d_2)

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        if d_2_arr is not None:
            for d_2 in d_2_arr:
                self.set_params({"d_2": d_2})
                man, exp = manexp10(d_2)
                ax.plot(  # type: ignore
                    h_0,
                    self.lamda(h_0) / (2 * np.pi),
                    label="$D_2 = {} \\times 10^{{{}}}$".format(man, exp),
                )
            legend = True
            loc = "upper left"
        else:
            ax.plot(h_0, self.lamda(h_0) / (2 * np.pi))  # type: ignore
            legend = False
        ax.set_xlabel(r"$\overline{H}_0$", fontdict=font)  # type: ignore
        ax.set_ylabel(r"$\lambda / 2\, \pi$", fontdict=font)  # type: ignore
        ax.set_xlim(h_0[0], h_0[-1])
        ax.set_ylim(0, get_ylim()[1])
        ax.set_xticks(np.arange(h_0[0], h_0[-1] + 1))
        ax.tick_params(axis="both", labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        if legend:
            ax.legend(loc=loc, fontsize=labelsize)  # type: ignore

        self.set_params({"d_2": store.d_2})  # type: ignore

        figdisp(fig, show, savepath, filename)

    def plot_b(
        self,
        h_0: npt.NDArray[np.float64],
        d_2_arr: npt.NDArray[np.float64] | None = None,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None,
    ):
        """
        Plot the imaginary quadratic parameter b as a function of the
        relative gain H_0.

        Parameters
        ----------
        h_0 : numpy.ndarray
            Array of relative gain values
        d_2_arr : numpy.ndarray
            Array of normalized dispersion parameter values;
            default = None (use the value from the object)
        show : bool
            If True, display the plot; default = True
        savepath : str
            Path to save the plot; default = None
        filename : str
            Name of the file to save the plot; default = None
        """
        self._check_h0(h_0)
        store = Storage(d_2=self._d_2)

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        if d_2_arr is not None:
            for d_2 in d_2_arr:
                self.set_params({"d_2": d_2})
                man, exp = manexp10(d_2)
                ax.plot(  # type: ignore
                    h_0,
                    self.b(h_0) / (2 * np.pi),
                    label="$D_2 = {} \\times 10^{{{}}}$".format(man, exp),
                )
            legend = True
            loc = "upper left"
        else:
            ax.plot(h_0, self.b(h_0))  # type: ignore
            legend = False
        ax.set_xlabel(r"$\overline{H}_0$", fontdict=font)  # type: ignore
        ax.set_ylabel(r"$b / 2 \pi$", fontdict=font)  # type: ignore
        ax.set_xlim(h_0[0], h_0[-1])
        ax.set_ylim(0, get_ylim()[1])
        ax.set_xticks(np.arange(h_0[0], h_0[-1] + 1))
        ax.tick_params(axis="both", labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        if legend:
            ax.legend(loc=loc, fontsize=labelsize)  # type: ignore

        self.set_params({"d_2": store.d_2})  # type: ignore

        figdisp(fig, show, savepath, filename)

    def plot_a(
        self,
        h_0: npt.NDArray[np.float64],
        d_2_arr: npt.NDArray[np.float64] | None = None,
        lef_arr: npt.NDArray[np.float64] | None = None,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None
    ):
        """
        Plot the real quadratic parameter a as a function of the
        relative gain H_0.

        Parameters
        ----------
        h_0 : numpy.ndarray
            Array of relative gain values
        d_2_arr : numpy.ndarray
            Array of normalized dispersion parameter values; default = None
        lef_arr : numpy.ndarray
            Array of linewidth enhancement factor values; default = None
        show : bool
            If True, display the plot; default = True
        savepath : str
            Path to save the plot; default = None
        filename : str
            Name of the file to save the plot; default = None

        Note that either d_2 or lef can be specified, but not both. If neither
        is specified, the values of d_2 and lef from the object will be used.
        """
        self._check_h0(h_0)
        store = Storage(d_2=self._d_2, lef=self._lef)

        if d_2_arr is not None and lef_arr is not None:
            raise ValueError(
                "Either d_2_arr or lef_arr mmay be specified, but not both"
            )

        fig, ax = plt.subplots(figsize=(8.0, 6.0))  # type: ignore
        if d_2_arr is not None:
            for d_2 in d_2_arr:
                self.set_params({"d_2": d_2})
                man, exp = manexp10(d_2)
                ax.plot(  # type: ignore
                    h_0,
                    self.a(h_0),
                    label="$D_2 = {} \\times 10^{{{}}}$".format(man, exp),
                )
            legend = True
            loc = "upper right"
        elif lef_arr is not None:
            for lef in lef_arr:
                self.set_params({"lef": lef})
                ax.plot(  # type: ignore
                    h_0,
                    self.a(h_0),
                    label="$\\alpha_\\mathrm{{LEF}} = {}$".format(lef),
                )
            legend = True
            loc = "upper left"
        else:
            ax.plot(h_0, self.a(h_0))  # type: ignore
            legend = False
        ax.set_xlabel(r"$\overline{H}_0$", fontdict=font)  # type: ignore
        ax.set_ylabel(r"$a$", fontdict=font)  # type: ignore
        ax.set_xlim(h_0[0], h_0[-1])
        ax.set_ylim(0, get_ylim()[1])
        ax.set_xticks(np.arange(h_0[0], h_0[-1] + 1))
        ax.tick_params(axis="both", labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        if legend:
            ax.legend(loc=loc, fontsize=labelsize)  # type: ignore

        self.set_params({"d_2": store.d_2, "lef": store.lef})  # type: ignore

        figdisp(fig, show, savepath, filename)
