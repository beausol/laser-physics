import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp

from typing import Any

from laser.utils import BaseClass, labelsize, font, get_ylim, figdisp
from rich import print


class ODEIntegrator(BaseClass):
    '''
    A virtual base-class wrapper for scipy.integrate.solve_ivp.

    Required Private Methods
    ------------------------
    _set_specs :
        Dict of keys that will be present in the parameter dictionary
        required by a derived class; default = dict()
    _deriv :
        Derivatrive function of (y, t) that follows scipy.integrate.solve_ivp;
        default: default: raises NotImplementedError
    _jac :
        (Optional) Jacobian function of (y, t) that follows
        scipy.integrate.solve_ivp; recommended for the Radau, BDF, and
        LSODA methods; if it's not implemented, then solve_ivp will use
        a finite difference approximation of the Jacobian matrix for
        these methods

    Public Methods
    --------------
    integrate :
       Using scipy.integrate.solve_ivp, numerically integrate variables
       defined in _deriv; reports and plots results if successful
    '''

    def _deriv(
        self,
        t: float,
        y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        '''
        Derivatrive function of (t, y) that follows scipy.integrate.solve_ivp.
        Required parameters should be private variables of the derived class,
        so that the solve_ivp 'args' parameter isn't needed. default: raises
        NotImplementedError

        Parameters
        ----------
        t : float64
            Current scalar value of time
        y : numpy.ndarray.float64 or numpy.ndarray.complex128
            Current values of the integration variables; typically,
            len(y) = number of integration variables

        Returns
        -------
        numpy.ndarray.float64
            Derivative of the integration variables with respect to time;
            the shape of the returned array should match the shape of y
        '''
        raise NotImplementedError

    def _report(self, sol: Any, elapsed: float, method: str,
                show: bool, savepath: str | None, filename: str | None):
        '''
        Report the results of self.integrate(...).

        Parameters
        ----------
        sol : bunch oblect
            Returned by scipy.integrate.solve_ivp
        elapsed : float64
            Total elapsed time in seconds for scipy.integrate.solve_ivp call
        method : string
            Integration method used by scipy.integrate.solve_ivp
        show : Boolean
            Show a brief summary of the integration process and the plot of the
            integration variables over time; default = True
        savepath : string
            Path of the folder/directory where a copy of the figure will be
            saved using Matplotlib's savefig; default = None
        filename : string
            Name of the file (including the format extension) that will
            contain the figure; default = None; note that both savepath and
            filename must be specified so that Matplotlib's savefig target --
            savepath + filename -- is valid
        '''
        print(sol.message)

        if sol.success:
            if show:
                if elapsed > 1.0:
                    elapsed = round(elapsed, 1)
                else:
                    elapsed = round(elapsed, 3)
                minutes, seconds = divmod(elapsed, 60)
                hours, minutes = divmod(minutes, 60)

                if hours:
                    print(
                        "Elapsed time: {} hours, {} minutes, {} seconds ({})"
                        .format(
                            int(hours),
                            int(minutes),
                            int(round(seconds)),
                            method
                        )
                    )
                elif minutes and not hours:
                    print(
                        "Elapsed time: {} minutes, {} seconds ({})".format(
                            int(minutes), int(round(seconds)), method
                        )
                    )
                else:
                    print(
                        "Elapsed time: {} seconds ({})"
                        .format(seconds, method)
                    )
                print(
                    (
                        "Derivative function calls: {} "
                        "({:.{prec}} calls/sec)"
                    ).format(
                        sol.nfev, float(sol.nfev)/elapsed, prec=3
                    )
                )
                if sol.njev > 0:
                    print(
                        "Jacobian function calls: {} ({:.{prec}} calls/sec)\n"
                        .format(sol.njev, float(sol.njev)/elapsed, prec=3)
                    )
                else:
                    print("\n")
                    
                self._simplot(sol, show, savepath, filename)

    def _simplot(
        self,
        sol: Any,
        show: bool,
        savepath: str | None,
        filename: str | None
    ):
        '''Plot the integration variables as a function of time.

        Parameters
        ----------
        sol : bunch oblect
            Returned by scipy.integrate.solve_ivp
        show : Boolean
            Show a plot of the integration variables over time; default = True
        savepath : string
            Path of the folder/directory where a copy of the figure will be
            saved using Matplotlib's savefig; default = None
        filename : string
            Name of the file (including the format extension) that will
            contain the figure; default = None; note that both savepath and
            filename must be specified so that Matplotlib's savefig target --
            savepath + filename -- is valid
        '''
        t = sol.t
        y = sol.y

        fig, ax = plt.subplots()  # type: ignore[assignment]
        ax.tick_params(axis='both', labelsize=labelsize)  # type: ignore
        ax.grid(visible=True)  # type: ignore
        ax.set_xlabel(r'$t$', fontdict=font)  # type: ignore
        ax.set_ylabel(r'$y(t)$', fontdict=font)  # type: ignore
        row, _ = y.shape
        for m in range(row):
            ax.plot(t, y[m, :], label=r'$y({})$'.format(m))  # type: ignore
        ax.legend(fontsize=labelsize)  # type: ignore
        ax.set_xlim(left=t[0], right=t[-1])
        ylim = get_ylim()
        ax.set_ylim(bottom=ylim[0], top=ylim[1])

        figdisp(fig, show, savepath, filename)

    def integrate(
        self,
        t_max: float,
        y0: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
        method: str = 'DOP853',
        npts: int | None = None,
        show: bool = True,
        savepath: str | None = None,
        filename: str | None = None
    ):
        '''Integrate an ODE using scipy.integrate.solve_ivp.

        Parameters
        ----------
        t_max : float64
            Final integration time (initial = 0.0)
        y0 : numpy.ndarray.float64 or numpy.ndarray.complex128
            Initial values of the integtation variables
        method : string
            Integration method used by scipy.integrate.solve_ivp;
            default = 'DOP853'
        npts : int32
            If npts is specified, then set scipy.integrate.solve_ivp's
            t_eval so that the integration variables are evaluated at npts + 1
            evenly spaced points from 0 to t_max; default = None
        show : Boolean
            Show a brief summary of the integration process and the plot of the
            integration variables over time; default = True
        savepath : string
            Path of the folder/directory where a copy of the figure will be
            saved using Matplotlib's savefig; default = None
        filename : string
            Name of the file (including the format extension) that will
            contain the figure; default = None; note that both savepath
            and filename must be specified so that Matplotlib's savefig
            target -- savepath + filename -- is valid

        Returns
        --------
        bunch object
            Returned by scipy.integrate.solve_ivp at the conclusion of the call

        Notes
        -----
        1. If the Jacobian function is not implemented, then the Radau, BDF,
           and LSODA methods will use a finite difference approximation of the
           Jacobian matrix; only the Radau method seems to do so accurately
        2. If the Jacobian function is implemented, then it should be named
           '_jac' and follow the scipy.integrate.solve_ivp convention
        '''
        t_span = (0, t_max)
        if npts is None:
            t_eval = None
        else:
            t_eval = np.linspace(t_span[0], t_span[1], num=npts, endpoint=True)

        jac = getattr(self, '_jac', None)
        elapsed: float = 0.0

        if method == 'RK23' or method == 'RK45' or method == 'DOP853':
            start = time.perf_counter()
            sol = solve_ivp(self._deriv, t_span, y0, method=method,
                            t_eval=t_eval)
            finish = time.perf_counter()
            elapsed = finish - start
        elif method == 'Radau' or method == 'BDF' or method == 'LSODA':
            start = time.perf_counter()
            sol = solve_ivp(self._deriv, t_span, y0, method=method,
                            jac=jac, t_eval=t_eval)
            finish = time.perf_counter()
            elapsed = finish - start
        else:
            raise ValueError(f"Unsupported integration method: {method}")

        self._report(sol, elapsed, method, show, savepath, filename)

        return sol
