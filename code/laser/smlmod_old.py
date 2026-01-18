import numpy as np
from scipy.special import erf
from scipy.integrate import solve_ivp, solve_bvp, simps
from scipy.optimize import brentq, fmin

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import animation

labelsize = 18
fontsize = 24
font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : fontsize,
        }
bbox = dict(boxstyle='round', edgecolor='#D7D7D7', facecolor='white', alpha=0.75)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')
plt.rc('animation', html='html5') # equivalent to rcParams['animation.html'] = 'html5'

from laser.utils import *
from numtools.odeintegrator import *

from enum import Enum, auto


class LineShape(object):
    def __init__(self, omega=0):
        self._omega = omega
        self.check_rho()

    def _lineshape(self, omega):
        raise NotImplementedError
    
    def get_omega(self):
        return self._omega

    def rho(self, omega):
        return np.real( self._lineshape(omega) )

    def iota(self, omega):
        return np.imag( self._lineshape(omega) )
    
    def check_rho(self):
        assert np.abs(self.rho(0.0) - 1.0) > np.finfo(float).eps, "Re(L(0)) = {}; must be 1.0.".format(self.rho(0.0))
        
class Lorentzian(LineShape):
    def _lineshape(self, omega):
        return 1 / (1 - 1j * omega)

class LaserAmplifierCW(object):
    def __init__(self, alpha_0, g_0, omega):

        '''Initialize a LaserAmplifierPulseProp object.
        
        Parameters
        ----------
        g_0 : numpy.float64
            Integrated single-pass unsaturated gain in the amplifier
        i_0 : numpy.float64
            Integrated input intensity (fluence)
        alpha_0 : numpy.float64
            Product of the absorption coefficient (in units of inverse length)
                and the amplifier length
        omega : numpy.float64
            Detuning of the laser carrier frequency with respect to the center
                frequency, normalized by the transverse decay time
        '''
        
        self._alpha_0 = alpha_0
        self._g_0 = g_0
        self._omega = omega

    def __str__(self):
        ''' Return a string containing the attributes of a LaserAmplifierPulseProp object.
            Example:
                model = LaserAmplifierCW(g_0, i_0, alpha_0, omega)
                print(pulse)
        '''
        template = ( "{}"
                     "\n"
                     "alpha_0 = {:.{prec}}; G_0 = {:.{prec}}; omega = {:.{prec}}"
                     "\n" )
        param_str = template.format(self.__class__.__name__, self._alpha_0, self._g_0, self._omega, prec = 3)
        
        return param_str

    def _set_i0(self, i_0):
        if i_0 is None or isinstance(i_0, np.ndarray):
            self._i_0 = i_0
        elif isinstance(i_0, list):
            self._i_0 = np.array(i_0)
        elif isinstance(i_0, float):
            self._i_0 = np.array([i_0])

    def _di_dz(self, z, iz):
            return self._g_0 * iz / (1 + self._omega**2 + iz) - self._alpha_0 * iz
    
    def _gain(self):
        return self._g_0 * (1 + self._omega**2) / (1 + self._omega**2 + self._iz)
 
    def set_vars(self, **kwargs):
        alpha_0 = kwargs.get('alpha_0')
        g_0 = kwargs.get('g_0')
        omega = kwargs.get('omega')
        if alpha_0 is None:
            pass
        else:
            self._alpha_0 = alpha_0
        if g_0 is None:
            pass
        else:
            self._g_0 = g_0
        if omega is None:
            pass
        else:
            self._omega = omega
    
    def get_igz(self):
        return self._iz, self._gz, self._z

    def solve(self, z, i_0):
        def funcz(iz, z, i_0):
            a0 = self._alpha_0
            g0 = self._g_0
            w2 = 1 + self._omega**2
            if a0 < np.finfo(float).eps:
                return np.log(iz/i_0) + ( (iz - i_0) - g0 * z ) / w2
            else:
                return ( w2 * np.log(iz/i_0) - (g0/a0) * np.log((g0 - a0 * (w2 + iz))/((g0 - a0 * (w2 + i_0))))
                        - (g0 - w2 * a0) * z )

        self._set_i0(i_0)
        iz = np.zeros((len(self._i_0), len(z)))
        iz[:,0] = self._i_0

        for m in range(len(self._i_0)):
            i0 = self._i_0[m]
            for n in range(len(z)-1):
                args = (z[n+1], i0)
                a, b = iz[m, n], iz[m, n] + self._di_dz(z[n], iz[m, n]) * 2 * (z[n + 1] - z[n])
                iz[m, n+1] = brentq(funcz, a, b, args)

        self._z = z
        self._iz = iz
        self._gz = self._gain()

    def integrate(self, z, i_0):
        self._set_i0(i_0)            
        sol = solve_ivp(self._di_dz, [0, 1], self._i_0, vectorized=True, dense_output=True)
        
        self._z = z
        self._iz = sol.sol(z)
        self._gz = self._gain()

    def plot_intensity(self, show=True, savepath=None):
        # labelsize = 18
        # fontsize = 24
        # font = {'family' : 'serif',
        #         'color'  : 'black',
        #         'weight' : 'normal',
        #         'size'   : fontsize,
        #         }
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')

        plt.figure(figsize=(8.0,6.0))
        plt.xlabel(r'$z$', fontdict=font)
        plt.ylabel(r'$G_\mathrm{eff}(z) \equiv I(z)/I_0$', fontdict=font)
        for n in range(self._iz.shape[0]):
            plt.plot(self._z, self._iz[n]/self._iz[n][0], label=r'$I_0 = {}$'.format(self._iz[n][0]))
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.grid(True)
        plt.legend(fontsize=labelsize)
        plt.xlim(self._z[0], self._z[-1])
        plt.ylim(get_ylim())

        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight')
            print("Saved {0}\n".format(savepath))
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_gain(self, show=True, savepath=None):
        # labelsize = 18
        # fontsize = 24
        # font = {'family' : 'serif',
        #         'color'  : 'black',
        #         'weight' : 'normal',
        #         'size'   : fontsize,
        #         }
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')

        plt.figure(figsize=(8.0,6.0))
        plt.xlabel(r'$z$', fontdict=font)
        plt.ylabel(r'$G(z)$', fontdict=font)
        for n in range(self._gz.shape[0]):
            plt.plot(self._z, self._gz[n], label=r'$I_0 = {}$'.format(self._iz[n][0]))
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.grid(True)
        plt.legend(loc='upper right', fontsize=labelsize)
        plt.xlim(self._z[0], self._z[-1])
        plt.ylim(0.0, get_ylim()[1])

        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight')
            print("Saved {0}\n".format(savepath))
        if show:
            plt.show()
        else:
            plt.close()


class Storage():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Config(Enum):
    URL = auto()
    SWL = auto()
#    SHB = auto()
    
    def kappa(self):
        if self.name == 'URL':
            return 1.0
        else:
            return 2.0
    
    def zlims(self):
        return (0.0, 1/self.kappa())


class Gain(object):
    def __init__(self, config:Config, gbar_0, **kwargs):
        self._config = config
        
        if abs(gbar_0) > np.finfo(float).eps:
            self._gbar_0 = gbar_0
        else:
            self._gbar_0 = 0.0
        self._z_1, self._z_2 = self._config.zlims()
        self._delta = 0.0
    
        self.set_args(**kwargs)
        
    #     # labelsize = 18
    #     # font
    # size = 24
    #     # font
    #  = {'family' : 'serif',
    #     #         'color'  : 'black',
    #     #         'weight' : 'normal',
    #     #         'size'   : font
    # size,
    #     #         }

    def __str__(self):
        ''' Return a string containing the attributes of a Gain object.
            Example:
                pulse = Gain(config, gbar_0, z_1, z_2, delta)
                print(pulse)
        '''
        template = ( "{}"
                     "\n"
                     "config = {}"
                     "\n"
                     "Gbar_0 = {:.{prec}}; z_1 = {:.{prec}}; z_2 = {:.{prec}}; delta = {:.{prec}}"
                     "\n" )
        param_str = template.format(self.__class__.__name__, self._config, self._gbar_0, self._z_1, self._z_2, self._delta, prec = 3)
        
        return param_str

    def set_args(self, **kwargs):
        z_1 = kwargs.get('z_1')
        if z_1 is None:
            pass
        else:
            self._z_1 = z_1
        z_2 = kwargs.get('z_2')
        if z_2 is None:
            pass
        else:
            self._z_2 = z_2

        delta = kwargs.get('delta')
        if delta is None:
            pass
        else:
            self._delta = delta

    def get_gbar_0(self):
        return self._gbar_0
    
    def set_gbar_0(self, gbar_0):
        self._gbar_0 = gbar_0
    
    def gain(self, z):
        raise NotImplementedError

    def intgain(self, z):
        gz = self.gain(z)
        igz = np.zeros_like(gz)
        for n in range(1, gz.size):
            igz[n] = simps(gz[0:n], x=z[0:n])
        return igz
    
    def check_config(self, config):
        assert self._config == config, "Incompatible configurations: {} and {}.".format(self._config.name, config.name)
        
    def plot_gain(self, z):
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')
        
        plt.figure(figsize=(8.0,6.0))
        plt.xlabel(r'$z$', fontdict=font)
        plt.ylabel(r'$G_0(z)$', fontdict=font)
        plt.plot(z, self.gain(z))
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.grid(True)
        plt.xlim(z[0], z[-1])
        plt.ylim(get_ylim())
        plt.show()

    def plot_intgain(self, z):
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')
        
        plt.figure(figsize=(8.0,6.0))
        plt.xlabel(r'$z$', fontdict=font)
        plt.ylabel(r'$\int_0^z d z^\prime\, G_0(z^\prime)$', fontdict=font)
        plt.plot(z, self.intgain(z), label='analytic')
        plt.plot(z, Gain.intgain(self, z), '--', label='numeric')
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.grid(True)
        plt.xlim(z[0], z[-1])
        plt.ylim(get_ylim())
        plt.legend(fontsize=labelsize)
        
        plt.show()

class GainFill(Gain):
    def gain(self, z):
        return self._gbar_0 * np.ones(z.shape)

    def intgain(self, z):
        return self._gbar_0 * z
        
class GainTrap(Gain):
    def gain(self, z):
        z_1 = self._z_1
        z_2 = self._z_2
        delta = self._delta
        
        g_max = self._gbar_0 / (self._config.kappa() * (z_2 - z_1))

        g = g_max * ( ( z - (z_1 - delta/2) ) /delta * np.heaviside(z - (z_1 - delta/2), 0.0) * np.heaviside(z_1 + delta/2 - z, 0.0)
                     + np.heaviside(z - (z_1 + delta/2), 1.0) * np.heaviside(z_2 - delta/2 - z, 1.0)
                     + ( 1 - (z - (z_2 - delta/2))/delta ) * np.heaviside(z - (z_2 - delta/2), 0.0) * np.heaviside(z_2 + delta/2 - z, 0.0) )

        return g

    def intgain(self, z):
        z_1 = self._z_1
        z_2 = self._z_2

        igz = self._gbar_0 * ( ((z - z_1) / (z_2 - z_1)) * np.heaviside(z - z_1, 1.0) * np.heaviside(z_2 - z, 1.0)
                            + np.heaviside(z - z_2, 0.0) ) / self._config.kappa()

        return igz


class LaserCW(object):
    '''I_out v. R for a couple of omegas
       I_out v. z
       Optimum output coupling for two omegas v. gain
       Optimum output intensity for two omegas v. gain
       Rigrod:
        I^\pm(z) for two omegas
        Sum of I's v z for two omegas
        I_out v. R (RR and approx) for two omegas
       AL:
        I^\pm(z) for two omegas
        Sum of I's v z for two omegas
        I_out v. R (AL and RR) for two omegas
       
        
    '''
    def __init__(self, config:Config, gainz:Gain, lossz:Gain, lineshape:LineShape, r_1, a_1, r_2, a_2):
        '''Initialize a LaserAmplifierPulseProp object.
        
        Parameters
        ----------
        alpha_0 : numpy.float64
            Product of the absorption coefficient (in units of inverse length)
                and the amplifier length
        omega_0 : numpy.float64
            Detuning of the laser carrier frequency with respect to the center
                frequency, normalized by the transverse decay time
        '''
        self._config = config
        #self._kappa = self._config.kappa()
        self._z_min, self._z_max = self._config.zlims()
        
        self.set_vars(
            gainz = gainz,
            lossz = lossz,
            lineshape = lineshape,
            r_1 = r_1,
            a_1 = a_1,
            r_2 = r_2,
            a_2 = a_2
        )
        
    #     # labelsize = 18
    #     # font
    # size = 24
    #     # font
    #  = {'family' : 'serif',
    #     #         'color'  : 'black',
    #     #         'weight' : 'normal',
    #     #         'size'   : font
    # size,
    #     #         }
    #     # bbox = dict(boxstyle='round', edgecolor='#D7D7D7', facecolor='white', alpha=0.75)
        
    def __str__(self):
        ''' Return a string containing the attributes of a LaserAmplifierPulseProp object.
            Example:
                pulse = LaserAmplifierPulseProp(g_0, i_0, tau_pulse)
                print(pulse)
        '''
        template = ( "{}"
                     "\n"
                     "config = {}"
                     "\n"
                     "alphabar_0 = {:.{prec}} ({:.{prec}} dB); Omega = {:.{prec}}; Gbar_0 = {:.{prec}}; Gbar_th = {:.{prec}}"
                     "\n"
                     "R_1 = {:.{prec}}; A_1 = {:.{prec}}; R_2 = {:.{prec}}; A_2 = {:.{prec}}"
                     "\n" )
       
        abar_0 = abs(self._lossz.get_gbar_0())
        omega = self._lineshape.get_omega()
        gbar_0 = self._gainz.get_gbar_0()
        gbar_th = self._threshold(self._lossz, self._omega, self._r_1 * self._r_2)
        retstr = template.format(self.__class__.__name__, self._config, abar_0, 10 * abar_0 / np.log(10.0), omega, gbar_0, gbar_th,
                                 self._r_1, self._a_1, self._r_2, self._a_2, prec = 3)
        retstr += self._gainz.__str__() + self._lossz.__str__()
        
        return retstr

    def _check_vars(self):
        self._gainz.check_config(self._config)
        self._lossz.check_config(self._config)
        self._lineshape.check_rho()

        assert self._r_1 > 0.0, "r_1 = {}; must be > 0.".format(self._r_1)
        assert self._r_1 < 1.0, "r_1 = {}; must be < 1.".format(self._r_1)
        assert self._a_1 >= 0.0, "a_1 = {}; must be >= 0.".format(self._a_1)
        assert self._r_1 + self._a_1 < 1.0, "r_1 + a_1 = {}; must be < 1.".format(self._r_1 + self._a_1)

        assert self._r_2 > 0.0, "r_2 = {}; must be > 0.".format(self._r_2)
        assert self._r_2 <= 1.0, "r_2 = {}; must be <= 1.".format(self._r_2)
        assert self._a_2 >= 0.0, "a_2 = {}; must be >= 0.".format(self._a_2)
        assert self._r_2 + self._a_2 <= 1.0, "r_2 + a_2 = {}; must be <= 1.".format(self._r_2 + self._a_2)

    def _threshold(self, lossz, omega, r):
        #return (1 + omega_0**2) * np.log( 1/(r * np.exp(lossz.get_gbar_0())) )
        return np.log( 1/(r * np.exp(lossz.get_gbar_0())) ) / self._lineshape.rho(omega)
    
    def _kappa(self, gbar_0, gbar_th, omega_0, r_1, r_2):
        return self._config.kappa()
    
    def _intensity_0(self, gbar_0, gbar_th, omega, kappa):
        #return (1 + omega_0**2) * (gbar_0 / gbar_th - 1) / kappa
        return (gbar_0 / gbar_th - 1) / ( kappa * self._lineshape.rho(omega) )
    
    def _below_threshold(self, gainz, lossz, omega, r):
        return gainz.get_gbar_0() <= self._threshold(lossz, omega, r)

    def _c2(self, r_1, r_2):
        return ( (r_1 * np.sqrt(r_2) * np.log(1.0/(r_1 * r_2)))
                  / ( (np.sqrt(r_1) + np.sqrt(r_2)) * (1 - np.sqrt(r_1 * r_2)) ) )
    
    def _kz(self, z):
        beta = np.log( 1/(self._r_1 * self._r_2 * np.exp(self._lossz.get_gbar_0())) ) / self._gainz.get_gbar_0()
        return np.exp( beta * self._gainz.intgain(z) + self._lossz.intgain(z) )
    
    def _grmg(self, gbar_0, npts):
        g_mg = np.tile(gbar_0, (npts, 1))
        
        r_mg = np.zeros_like(g_mg)
        for n in range(len(gbar_0)):
            r_min = np.exp(-gbar_0[n] / (1 + self._omega**2) - self._lossz.get_gbar_0()) + np.finfo(float).eps
            r_max = 1.0 - self._a_1 - 10 * np.finfo(float).eps
            r = np.linspace(r_min, r_max, npts, endpoint=True)
            r_mg[:,n] = r
        
        return (r_mg, g_mg)

    def _funcr(self, r, g):
        raise NotImplementedError

    def set_vars(self, check=True, **kwargs):
        gainz = kwargs.get('gainz')
        if gainz is None:
            pass
        else:
            self._gainz = gainz
        gbar_0 = kwargs.get('gbar_0')
        if gbar_0 is None:
            pass
        else:
            self._gainz.set_gbar_0(gbar_0)

        lossz = kwargs.get('lossz')
        if lossz is None:
            pass
        else:
            self._lossz = lossz
        abar_0 = kwargs.get('abar_0')
        if abar_0 is None:
            pass
        else:
            if abs(abar_0) > np.finfo(float).eps:
                self._lossz.set_gbar_0(-abs(abar_0))
            else:
                self._lossz.set_gbar_0(0.0)

        omega = kwargs.get('omega')
        if omega is None:
            pass
        else:
            self._omega = omega
        
        r_1 = kwargs.get('r_1')
        if r_1 is None:
            pass
        else:
            self._r_1 = r_1
        a_1 = kwargs.get('a_1')
        if a_1 is None:
            pass
        else:
            self._a_1 = a_1

        r_2 = kwargs.get('r_2')
        if r_2 is None:
            pass
        else:
            self._r_2 = r_2
        a_2 = kwargs.get('a_2')
        if a_2 is None:
            pass
        else:
            self._a_2 = a_2
        
        if check:
            self._check_vars()

    def approx_iz(self, z):
        if self._below_threshold(self._gainz, self._lossz, self._omega, self._r_1 * self._r_2):
            ip_z = np.zeros_like(z)
            im_z = np.zeros_like(z)
        else:
            c2 = self._c2(self._r_1, self._r_2)
            kz = self._kz(z)
            up_0 = c2 * kz
            um_0 = c2 / (self._r_1 * kz)

            gbar_0 = self._gainz.get_gbar_0()
            gbar_th = self._threshold(self._lossz, self._omega, self._r_1 * self._r_2)
            kappa = self._kappa(gbar_0, gbar_th, self._omega, self._r_1, self._r_2)
            i_0 = self._intensity_0(gbar_0, gbar_th, self._omega, kappa)

            ip_z = i_0 * up_0
            im_z = i_0 * um_0

        return ip_z, im_z

    def approx_io(self, gbar_0, r):
        gbar_th = self._threshold(self._lossz, self._omega, r)
        kappa = self._kappa(gbar_0, gbar_th, self._omega, r, np.ones_like(r))
        i_0 = self._intensity_0(gbar_0, gbar_th, self._omega, kappa)
        c2 = self._c2(r, np.ones_like(r))

        out = ((1.0 - self._a_1 - r) * c2 / r) * i_0

        if isinstance(out, np.ndarray):
            out[out < 0] = -1
        else:
            if out < 0:
                out = -1

        return out
    
    def approx_opt(self, g_max, npts):
        gp2 = np.exp(self._lossz.get_gbar_0())
        loss = -np.log((1 - self._a_1) * gp2)
        g_min = (1 + self._omega**2) * loss + np.finfo(float).eps
        gbar_0 = np.linspace(g_min, g_max, npts, endpoint=True)
        
        r_opt = np.exp(-np.sqrt( (gbar_0/(1 + self._omega**2)) * loss )) / gp2
        #i_opt = (1 + self._omega_0**2) * ( np.sqrt(gbar_0/(1 + self._omega_0**2)) - np.sqrt(loss) )**2
        i_opt = self.approx_io(gbar_0, r_opt)
        
        return gbar_0, r_opt, i_opt

    def solve_iz(self, z):
        raise NotImplementedError

    def solve_io(self, gbar_0, r):
        raise NotImplementedError

    def integrate_iz(self, z, verbose=0):
        iz = np.zeros((2, z.size))
        iz[0], iz[1] = self.approx_iz(z)
        
        sol = solve_bvp(self._di_dz, self._bc, z, iz, verbose=verbose)
        if verbose != 0:
            print("\n")
        return sol

    def integrate_io(self, gbar_0, r):
        raise NotImplementedError
    
    def exact_opt(self, g_max, npts):
        gbar_0, r_x_opt, i_x_opt = self.approx_opt(g_max, npts)
        
        r_e_opt = np.zeros_like(r_x_opt)
        i_e_opt = np.zeros_like(i_x_opt)
        
        for m in np.ndindex(gbar_0.shape):
            sol = fmin(self._funcr, r_x_opt[m], args=(gbar_0[m],), full_output=True, disp=False)
            r_e_opt[m] = sol[0]
            i_e_opt[m] = -sol[1]
        
        return r_e_opt, i_e_opt

    def plot_r(self, gbar_0, npts=101, show=True, savepath=None):
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')
        
        r_mg, g_mg = self._grmg(gbar_0, npts)
        i_x = self.approx_io(g_mg, r_mg)
        i_e, label_e = self._exact_io(g_mg, r_mg)

        annotation = ( r'$\overline{{\alpha}}_0 = {:.{prec}}$'.format(abs(self._lossz.get_gbar_0()), prec=2) + '\n'
                     + r'$\Omega = {}$'.format(self._omega) +'\n'
                     + r'$A = {}$'.format(self._a_1) )

        plt.figure(figsize=(8.0,6.0))
        plt.xlabel(r'$R$', fontdict=font)
        plt.ylabel(r'$I_\mathrm{out}$', fontdict=font)
        for n in range(gbar_0.size):
            plt.plot(r_mg[:,n], i_e[:,n], label=r'$\overline{{G}}_0 = {}$'.format(gbar_0[n]) + label_e)
            plt.plot(r_mg[:,n], i_x[:,n], '--', label=r'$\overline{{G}}_0 = {}$: approximate'.format(gbar_0[n]))
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.grid(True)
        plt.xlim(0, 1)
        y_max = get_ylim()[1]
        plt.ylim(np.finfo(float).eps, y_max)
        plt.legend(fontsize=labelsize, loc='upper left')
        plt.text(0.76, 0.75 * y_max, annotation, bbox=bbox, fontdict=font)

        if savepath is None:
            pass
        else:
            plt.savefig(savepath, bbox_inches='tight')
            print("Saved {0}\n".format(savepath))
        if show:
            plt.show()
        else:
            plt.close()

    def plot_opt(self, g_max, npts=101, show=True, savepath=None):
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')
        
        gbar_0, r_x_opt, i_x_opt = self.approx_opt(g_max, npts)
        r_e_opt, i_e_opt = self.exact_opt(g_max, npts)
        
        annotation = (  r'$\overline{{\alpha}}_0 = {:.{prec}}$'.format(abs(self._lossz.get_gbar_0()), prec=2) + '\n'
                      + r'$\Omega = {}$'.format(self._omega) + '\n'
                      + r'$A = {}$'.format(self._a_1) )
        
        plt.figure(figsize=(8.0,6.0))
        plt.xlabel(r'$\overline{{G}}_0$', fontdict=font)
        plt.ylabel(r'$R_\mathrm{opt}$', fontdict=font)
        plt.plot(gbar_0, r_e_opt, label='exact')
        plt.plot(gbar_0, r_x_opt, '--', label='approx')
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.grid(True)
        plt.legend(fontsize=labelsize)
        plt.xlim(0, g_max)
        (y_min, y_max) = get_ylim()
        if y_max > 1.0:
            y_max = 1.0
        plt.ylim(y_min, y_max)
        plt.text(0.05*g_max, y_min + 0.05*(y_max - y_min), annotation, bbox=bbox, fontdict=font)

        if savepath is None:
            pass
        else:
            idx = savepath.index('.pdf')
            savepath_r = savepath[:idx] + '_r' + savepath[idx:]
            plt.savefig(savepath_r, bbox_inches='tight')
            print("Saved {0}\n".format(savepath_r))
        if show:
            plt.show()
        else:
            plt.close()

        plt.figure(figsize=(8.0,6.0))
        plt.xlabel(r'$\overline{{G}}_0$', fontdict=font)
        plt.ylabel(r'$I_\mathrm{opt}$', fontdict=font)
        plt.plot(gbar_0, i_e_opt, label='exact')
        plt.plot(gbar_0, i_x_opt, '--', label='approx')
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.grid(True)
        plt.legend(fontsize=labelsize)
        plt.xlim(0, g_max)
        y_max = get_ylim()[1]
        plt.ylim(0, y_max)
        plt.text(0.75*g_max, 0.05*y_max, annotation, bbox=bbox, fontdict=font)

        if savepath is None:
            pass
        else:
            idx = savepath.index('.pdf')
            savepath_i = savepath[:idx] + '_i' + savepath[idx:]
            plt.savefig(savepath_i, bbox_inches='tight')
            print("Saved {0}\n".format(savepath_i))
        if show:
            plt.show()
        else:
            plt.close()

class LaserCWURL(LaserCW):
    def __init__(self, gainz, lossz, omega, r, a):
        LaserCW.__init__(self, Config.URL, gainz, lossz, omega, r, a, 1.0, 0.0)
        
    def _di_dz(self, z, iz):
        dip_dz =  self._gainz.gain(z) * iz[0] / ( 1 + self._omega**2 + iz[0] ) + self._lossz.gain(z) * iz[0]
        dim_dz = -self._gainz.gain(z) * iz[1] / ( 1 + self._omega**2 + iz[1] ) - self._lossz.gain(z) * iz[1]
        return np.vstack((dip_dz, dim_dz))

    def _bc(self, ia, ib):
        return np.array([ia[0] - self._r_1 * ib[0], ib[1] - self._r_1 * ia[1]])

    def _exact_io(self, g_mg, r_mg):
        return ( self.solve_io(g_mg, r_mg), ': solve (brentq)' )

    def _funcr(self, r, g):
        return -self.solve_io(np.array([g]), r)[0]
    
    def set_vars(self, **kwargs):
        r = kwargs.get('r')
        if r is None:
            pass
        else:
            self._r_1 = r
        a = kwargs.get('a')
        if a is None:
            pass
        else:
            self._a_1 = a

        LaserCW.set_vars(self, **kwargs)

    def solve_iz(self, z):
        def funcz(iz, z, i0, gbar_0, abar_0, w2):
            if abar_0 < np.finfo(float).eps:
                return np.log(iz/i0) + ( (iz - i0) - gbar_0 * z ) / w2
            else:
                return ( w2 * np.log(iz/i0) - (gbar_0/abar_0) * np.log((gbar_0 - abar_0 * (w2 + iz))/((gbar_0 - abar_0 * (w2 + i0))))
                        - (gbar_0 - w2 * abar_0) * z )

        def dip_dz(iz, gbar_0, abar_0, w2):
            return gbar_0 * iz / (w2 + iz) - abar_0 * iz

        iz = np.zeros_like(z)
        if self._below_threshold(self._gainz, self._lossz, self._omega, self._r_1 * self._r_2):
            return iz

        gbar_0 = self._gainz.get_gbar_0()
        abar_0 = abs(self._lossz.get_gbar_0())
        w2 = 1 + self._omega**2
        gbar_th = self._threshold(self._lossz, self._omega, self._r_1 * self._r_2)

        if abar_0 < np.finfo(float).eps:
            i_1 = (gbar_0 - gbar_th) / (1 - self._r_1)    
        else:
            x = (abar_0/gbar_0) * ( gbar_0 - gbar_th)
            i_1 = (gbar_0/abar_0 - w2) * (1 - np.exp(-x)) / (1 - self._r_1 * np.exp(-x))
        i_0 = self._r_1 * i_1

        iz[0] = i_0
        iz[-1] = i_1

        for n in range(len(z)-2):
            args = (z[n+1], i_0, gbar_0, abar_0, w2)
            a, b = iz[n], iz[n] + dip_dz(iz[n], gbar_0, abar_0, w2) * 2 * (z[n + 1] - z[n])
            iz[n+1] = brentq(funcz, a, b, args)

        return iz

    def solve_io(self, g0, r):
        gt = self._threshold(self._lossz, self._omega, r)
        abar_0 = abs(self._lossz.get_gbar_0())
        w2 = 1 + self._omega**2
        if abar_0 < np.finfo(float).eps:
            i_1 = (g0 - gt) / (1 - r)    
        else:
            x = (abar_0/g0) * ( g0 - gt)
            i_1 = (g0/abar_0 - w2) * (1 - np.exp(-x)) / (1 - r * np.exp(-x))
        out = (1 - self._a_1 - r) * i_1

        if isinstance(out, np.ndarray):
            out[out < 0] = -1
        else:
            if out < 0:
                out = -1

        return out

    def plot_z(self, npts=101, bpts=15, show=True, savepath=None):
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')
        
        z_bvp = np.linspace(0, 1.0, bpts, endpoint=True)
        bvp = self.integrate_iz(z_bvp, verbose=2)

        z = np.linspace(0, 1.0, npts, endpoint=True)
        ip_i = bvp.sol(z)[0]
        ip_s = self.solve_iz(z)
        ip_x = self.approx_iz(z)[0]
        
        annotation = ( r'$\overline{{G}}_0 = {:.{prec}}$'.format(self._gainz.get_gbar_0(), prec=3) +'\n'
                     + r'$\overline{{\alpha}}_0 = {:.{prec}}$'.format(abs(self._lossz.get_gbar_0()), prec=2) + '\n'
                     + r'$\Omega = {}$'.format(self._omega) +'\n'
                     + r'$R = {}$'.format(self._r_1) )
        
        plt.figure(figsize=(8.0,6.0))
        plt.xlabel(r'$z$', fontdict=font)
        plt.ylabel(r'$I(z)$', fontdict=font)
        plt.plot(z, ip_i, label='integrate (bvp)')
        plt.plot(z, ip_s, '--', label='solve (brentq)')
        plt.plot(z, ip_x, label='approximate')
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.grid(True)
        plt.legend(fontsize=labelsize)
        plt.xlim(z[0], z[-1])
        (y_min, y_max) = get_ylim()
        plt.ylim(y_min, y_max)
        plt.text(0.75*(z[-1]-z[0]), y_min + 0.05*(y_max - y_min), annotation, bbox=bbox, fontdict=font)

        if savepath is None:
            pass
        else:
            plt.savefig(savepath, bbox_inches='tight')
            print("Saved {0}\n".format(savepath))
        if show:
            plt.show()
        else:
            plt.close()

class LaserCWSWL(LaserCW):
    def __init__(self, gainz, lossz, omega, r_1, a_1, r_2, a_2):
        LaserCW.__init__(self, Config.SWL, gainz, lossz, omega, r_1, a_1, r_2, a_2)
            
    def _phi_rr(self, gainz, omega, r_1, r_2):
        lossz = GainFill(Config.SWL, 0)
        gbar_rr = self._threshold(lossz, omega, r_1 * r_2)
        gbar_0 = gainz.get_gbar_0()
        kappa = self._kappa(gbar_0, gbar_rr, omega, r_1, r_2)
        i_0 = self._intensity_0(gbar_0, gbar_rr, omega, kappa)
        
        return (self._c2(r_1, r_2)/np.sqrt(r_1)) * i_0
    
    def _di_dz(self, z, iz):
        dip_dz =  self._gainz.gain(z) * iz[0] / ( 1 + self._omega**2 + iz[0] + iz[1]) + self._lossz.gain(z) * iz[0]
        dim_dz = -self._gainz.gain(z) * iz[1] / ( 1 + self._omega**2 + iz[0] + iz[1] ) - self._lossz.gain(z) * iz[1]
        return np.vstack((dip_dz, dim_dz))

    def _bc(self, ia, ib):
        return np.array([ia[0] - self._r_1 * ia[1], ib[1] - self._r_2 * ib[0]])

    def _exact_io(self, g_mg, r_mg):
        return ( self.integrate_io(g_mg, r_mg), ': integrate (bvp)' )

    def _funcr(self, r, g):
        return -self.integrate_io(np.array([g]), r)[0]

    def store_vars(self, **kwargs):
        return Storage(**kwargs)
    
    def solve_iz(self, z):
        def funcz(iz, z, i0, gbar_0, w2, phi_rr):
            return w2 * np.log(iz/i0) +  (iz - i0) * (1 + phi_rr**2 / (iz * i0)) - gbar_0 * z

        def dip_dz(iz, gbar_0, w2, phi_rr):
            return gbar_0 * iz / (w2 + iz + phi_rr**2 / iz)

        iz = np.zeros_like(z)
        if self._below_threshold(self._gainz, self._lossz, self._omega, self._r_1 * self._r_2):
            return iz, iz

        gbar_0 = self._gainz.get_gbar_0()
        w2 = 1 + self._omega**2
        phi_rr = self._phi_rr(self._gainz, self._omega, self._r_1, self._r_2)

        i_0 = phi_rr * np.sqrt(self._r_1)
        i_1 = phi_rr / np.sqrt(self._r_2)

        iz[0] = i_0
        iz[-1] = i_1
        for n in range(len(z)-2):
            args = (z[n+1], i_0, gbar_0, w2, phi_rr)
            a, b = iz[n], iz[n] + dip_dz(iz[n], gbar_0, w2, phi_rr) * 2 * (z[n + 1] - z[n])
            iz[n+1] = brentq(funcz, a, b, args)

        return iz, phi_rr**2 / iz

    def integrate_io(self, gbar_0, r):
        storage = self.store_vars(gbar_0=self._gainz.get_gbar_0(), r_1=self._r_1, r_2=self._r_2, a_2=self._a_2)
        self.set_vars(r_2 = 1.0, a_2 = 0.0)
        
        bpts = 15
        z_bvp = np.linspace(self._z_min, self._z_max, bpts, endpoint=True)

        im_out = np.zeros_like(gbar_0)
        for m in np.ndindex(gbar_0.shape):
            self.set_vars(check=False, gbar_0 = gbar_0[m], r_1 = r[m])
            bvp = self.integrate_iz(z_bvp)
            ipm = bvp.sol(z_bvp)
            im_out[m] = ipm[1][0] * (1 - self._a_1 - r[m])
        
        self.set_vars(gbar_0 = storage.gbar_0, r_1 = storage.r_1, r_2 = storage.r_2, a_2 = storage.a_2)

        return im_out

    # def exact_opt(self, g_max, npts):
    #     def funcr(r, g):
    #         return -self.integrate_io(np.array([g]), r)[0]
        
    #     gbar_0, r_x_opt, i_x_opt = self.approx_opt(g_max, npts)
        
    #     r_e_opt = np.zeros_like(r_x_opt)
    #     i_e_opt = np.zeros_like(i_x_opt)
        
    #     for m in np.ndindex(gbar_0.shape):
    #         sol = fmin(funcr, r_x_opt[m], args=(gbar_0[m],), full_output=True, disp=False)
    #         r_e_opt[m] = sol[0]
    #         i_e_opt[m] = -sol[1]
        
    #     return r_e_opt, i_e_opt

    # def exact_opt(self, gbar_0):
    #     npts = gbar_0.size
    #     r = np.linspace(np.finfo(float).eps, 1.0 - self._a_1, npts, endpoint=True)
    #     r_mg, g_mg = np.meshgrid(r, gbar_0, sparse=False, indexing='ij')

    #     ip_s = self.integrate_io(g_mg, r_mg)
    #     indx_opt = np.argmax(ip_s, axis=0)
    #     indx_opt[0] = npts - 1
    #     r_e_opt = r[indx_opt]
    #     i_e_opt = np.amax(ip_s, axis=0)
        
    #     return r_e_opt, i_e_opt

    def plot_z(self, npts=101, bpts=15, show=True, savepath=None):
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')
        
        z_bvp = np.linspace(0, self._z_max, bpts, endpoint=True)
        bvp = self.integrate_iz(z_bvp, verbose=2)

        z = np.linspace(self._z_min, self._z_max, npts, endpoint=True)
        ip_i = bvp.sol(z)
        ip_s = self.solve_iz(z)
        ip_x = self.approx_iz(z)
        
        annotation = ( r'$\overline{{G}}_0 = {:.{prec}}$'.format(self._gainz.get_gbar_0(), prec=3) + '\n'
                     + r'$\overline{{\alpha}}_0 = {:.{prec}}$'.format(abs(self._lossz.get_gbar_0()), prec=2) + '\n'
                     + r'$\Omega = {}$'.format(self._omega) + '\n'
                     + r'$R_1 = {:.{prec}}$'.format(self._r_1, prec=3) + '\n'
                     + r'$R_2 = {:.{prec}}$'.format(self._r_2, prec=3) )
        
        plt.figure(figsize=(8.0,6.0))
        plt.xlabel(r'$z$', fontdict=font)
        plt.ylabel(r'$I^\pm(z)$', fontdict=font)
        plt.plot(z, ip_i[0], label=r'$I^+(z)$' + ': integrate (bvp)')
        plt.plot(z, ip_s[0], '-.', label=r'$I^+(z)$' + ': solve (brentq)')
        plt.plot(z, ip_x[0], '--', label=r'$I^+(z)$' + ': approximate')
        plt.plot(z, ip_i[1], label=r'$I^-(z)$' + ': integrate (bvp)')
        plt.plot(z, ip_s[1], '-.', label=r'$I^-(z)$' + ': solve (brentq)')
        plt.plot(z, ip_x[1], '--', label=r'$I^-(z)$' + ': approximate')
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.grid(True)
        plt.legend(fontsize=labelsize, loc='upper left')
        plt.xlim(z[0], z[-1])
        (y_min, y_max) = get_ylim()
        plt.ylim(y_min, y_max)
        plt.text(0.75*(z[-1]-z[0]), y_min + 0.06*(y_max - y_min), annotation, bbox=bbox, fontdict=font)

        if savepath is None:
            pass
        else:
            plt.savefig(savepath, bbox_inches='tight')
            print("Saved {0}\n".format(savepath))
        if show:
            plt.show()
        else:
            plt.close()

    def plot_sum(self, npts=101, bpts=15, show=True, savepath=None):
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')
        
        z_bvp = np.linspace(self._z_min, self._z_max, bpts, endpoint=True)
        bvp = self.integrate_iz(z_bvp)

        z = np.linspace(self._z_min, self._z_max, npts, endpoint=True)
        ip_i = bvp.sol(z)
        ip_s = self.solve_iz(z)
        ip_x = self.approx_iz(z)
        
        annotation = ( r'$\overline{{G}}_0 = {:.{prec}}$'.format(self._gainz.get_gbar_0(), prec=3) + '\n'
                     + r'$\overline{{\alpha}}_0 = {:.{prec}}$'.format(abs(self._lossz.get_gbar_0()), prec=2) + '\n'
                     + r'$\Omega = {}$'.format(self._omega) + '\n'
                     + r'$R_1 = {:.{prec}}$'.format(self._r_1, prec=3) + '\n'
                     + r'$R_2 = {:.{prec}}$'.format(self._r_2, prec=3) )
        
        plt.figure(figsize=(8.0,6.0))
        plt.xlabel(r'$z$', fontdict=font)
        plt.ylabel(r'$I^+(z) + I^-(z)$', fontdict=font)
        plt.plot(z, ip_i[0] + ip_i[1], label='integrate (bvp)')
        plt.plot(z, ip_x[0] + ip_x[1], '--', label='approximate')
        plt.plot(z, np.mean(ip_i[0] + ip_i[1]) * np.ones_like(z), '-.', label='average (bvp)')
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.grid(True)
        plt.legend(fontsize=labelsize, loc='lower left')
        plt.xlim(z[0], z[-1])
        (y_min, y_max) = get_ylim()
        plt.ylim(0, y_max)
        plt.text(0.75*(z[-1]-z[0]), 0.06*y_max, annotation, bbox=bbox, fontdict=font)

        if savepath is None:
            pass
        else:
            plt.savefig(savepath, bbox_inches='tight')
            print("Saved {0}\n".format(savepath))
        if show:
            plt.show()
        else:
            plt.close()

class LaserCWSHB(LaserCWSWL):
    def _phi(self, gbar_0, gbar_th, omega, r_1, r_2):
        i_0 = self._intensity_0(gbar_0, gbar_th, omega, self._config.kappa())
        phi_0 = (self._c2(r_1, r_2)/np.sqrt(r_1)) * i_0 / (1 + omega**2)
        
        beta = (r_1 + r_2) * (1 + np.sqrt(r_1*r_2)) / ( np.sqrt(r_1*r_2) * (np.sqrt(r_1) + np.sqrt(r_2)) )
        phi = (3 / (2 * beta)) * ( np.sqrt(1 + (8*beta/9) * phi_0) - 1 )
        
        return (phi, phi_0)
    
    def _kappa(self, gbar_0, gbar_th, omega, r_1, r_2):
        phi, phi_0 = self._phi(gbar_0, gbar_th, omega, r_1, r_2)

        return 2 + phi / (phi_0 - phi/2)

    def _di_dz(self, z, iz):
        a = 1 + self._omega**2 + iz[0] + iz[1]
        b = 2 * np.sqrt( iz[0] * iz[1] )
    
        dip_dz =  self._gainz.gain(z) * ( (iz[0] - a/2) / np.sqrt(a**2 - b**2) + 0.5 ) + self._lossz.gain(z) * iz[0]
        dim_dz = -self._gainz.gain(z) * ( (iz[1] - a/2) / np.sqrt(a**2 - b**2) + 0.5 ) - self._lossz.gain(z) * iz[1]

        return np.vstack((dip_dz, dim_dz))

    def solve_iz(self, z):
        def intensity_1(phi):
            return (phi * np.sqrt(phi**2 * (1.0 - self._r_1)**2 + 4.0*self._r_1) - phi**2 * (1.0 - self._r_1))/2.0

        def intensity_2(phi):
            return (phi * np.sqrt(phi**2 * (1.0/self._r_2 - 1.0)**2 + 4.0/self._r_2) + phi**2 * (1.0/self._r_2 - 1.0))/2.0

        def funcp(phi):
            return ( np.log(intensity_2(phi)/intensity_1(phi))
                    + (intensity_2(phi) - intensity_1(phi))
                    + phi**2 * (1.0/intensity_1(phi) - 1.0/intensity_2(phi))
                    - 0.5 * self._gainz.get_gbar_0() / (1 + self._omega**2) )
        
        def funcz(iz, z, i0, gbar_0, w2, phi):
            return np.log(iz/i0) + (iz - i0) + phi**2 * (1.0/i0 - 1.0/iz) - gbar_0 * z / w2

        def dip_dz(iz, gbar_0, w2, phi):
            a = w2 + iz + phi**2 / iz
            b = 2 * phi
            return gbar_0 * ( (iz - a/2) / np.sqrt(a**2 - b**2) + 0.5 )

        gbar_0 = self._gainz.get_gbar_0()
        gbar_th = self._threshold(self._lossz, self._omega, self._r_1 * self._r_2)
        w2 = 1 + self._omega**2
        
        phi_x = self._phi(gbar_0, gbar_th, self._omega, self._r_1, self._r_2)[0]
        band = 0.5
        a = phi_x - band
        b = phi_x + band
        phi = brentq(funcp, a, b)
        i_1 = intensity_1(phi)
        i_2 = intensity_2(phi)
        
        iz = np.zeros(len(z))
        iz[0] = i_1
        iz[-1] = i_2
        for n in range(len(z)-2):
            args = (z[n+1], i_1, gbar_0, w2, phi)
            a, b = iz[n], iz[n] + dip_dz(iz[n], gbar_0, w2, phi) * 2 * (z[n + 1] - z[n])
            iz[n+1] = brentq(funcz, a, b, args)

        ir = (1 + self._omega**2) * (iz + phi**2)
        il = (1 + self._omega**2) * (phi**2/iz + phi**2)    
    
        return ir, il
    
    def exact_opt(self, g_max, npts):
        def funcr(r, g):
            return -self.integrate_io(np.array([g]), r)[0]
        
        gbar_0, r_x_opt, i_x_opt = self.approx_opt(g_max, npts)
        
        r_e_opt = np.zeros_like(r_x_opt)
        i_e_opt = np.zeros_like(i_x_opt)
        
        for m in np.ndindex(gbar_0.shape):
            sol = fmin(funcr, r_x_opt[m], args=(gbar_0[m],), full_output=True, disp=False)
            r_e_opt[m] = sol[0]
            i_e_opt[m] = -sol[1]
        
        return r_e_opt, i_e_opt

    def plot_k(self, gbar_0, npts=101, show=True, savepath=None):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')

        r_mg, g_mg = self._grmg(gbar_0, npts)
        
        gbar_th = self._threshold(self._lossz, self._omega, r_mg)
        kappa = self._kappa(g_mg, gbar_th, self._omega, r_mg, 1.0)
        kappa[0,:] = 3.0

        annotation = ( r'$\overline{{\alpha}}_0 = {:.{prec}}$'.format(abs(self._lossz.get_gbar_0()), prec=2) + '\n'
                     + r'$\Omega = {}$'.format(self._omega) )
        
        plt.figure(figsize=(8.0,6.0))
        plt.xlabel(r'$R$', fontdict=font)
        plt.ylabel(r'$\kappa$', fontdict=font)
        for n in range(gbar_0.size):
            plt.plot(r_mg[:,n], kappa[:,n], label=r'$\overline{{G}}_0 = {}$'.format(gbar_0[n]))
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.grid(True)
        plt.xlim(0.0, 1.0)
        plt.ylim(2.0, 3.0)
        plt.legend(fontsize=labelsize, loc='lower left')
        plt.text(0.035, 2.5, annotation, bbox=bbox, fontdict=font)

        if savepath is None:
            pass
        else:
            plt.savefig(savepath, bbox_inches='tight')
            print("Saved {0}\n".format(savepath))
        if show:
            plt.show()
        else:
            plt.close()


class LaserAmplifierPulseProp(object):
    def __init__(self, g_0, i_0, tau_pulse):

        '''Initialize a LaserAmplifierPulseProp object.
        
        Parameters
        ----------
        g_0 : numpy.float64
            Integrated single-pass unsaturated gain in the amplifier
        i_0 : numpy.float64
            Integrated input intensity (fluence)
        tau_pulse : numpy.float64
            Pulse width (in units of the end-to-end group propagation time through the amplifier)
        '''
        
        self._g_0 = g_0
        self._i_0 = i_0
        self._tau_pulse = tau_pulse

    def __str__(self):
        ''' Return a string containing the attributes of a LaserAmplifierPulseProp object.
            Example:
                pulse = LaserAmplifierPulseProp(g_0, i_0, tau_pulse)
                print(pulse)
        '''
        template = ( "{}"
                     "\n"
                     "I_0 = {:.{prec}}; G_0 = {:.{prec}}; tau_pulse = {:.{prec}}"
                     "\n" )
        param_str = template.format(self.__class__.__name__, self._i_0, self._g_0, self._tau_pulse, prec = 3)
        
        return param_str

    def _it(self, t):
        '''Compute the current pulse intensity as a function of time.
        '''
        raise NotImplementedError

    def _jt(self, t):
        '''Compute the current pulse fluence (the integrated pulse intensity)
           as a function of time.
        '''
        raise NotImplementedError
    
    def _gz(self, z):
        '''Compute the current gain profile as a function of position within
           the amplifier (0 < z < 1)
        '''
        raise NotImplementedError

    def _hz(self, z):
        '''Compute the current exponential integrated gain as a function of
           position within the amplifier (0 < z < 1)
        '''
        raise NotImplementedError

    def get_keywords(self):
        ''' Return a string containing the attributes of a LaserAmplifierPulseProp object in a simple form
            that can be included in the Keywords field of a PDF document's Properties.
        '''
        template = ( "{}: I_0 = {:.{prec}}; G_0 = {:.{prec}}; tau_pulse = {:.{prec}}" )
        keyword_str = template.format(self.__class__.__name__, self._i_0, self._g_0, self._tau_pulse, prec = 3)

        return keyword_str

    def intensity(self, z, t):
        it = self._it(t - z)
        jt = self._jt(t - z)
        hz = self._hz(z)

        return it / ( 1 + (1/hz - 1) * np.exp(-jt) )

    def gain(self, z, t):
        jt = self._jt(t - z)
        gz = self._gz(z)
        hz = self._hz(z)

        return gz / ( 1 + (np.exp(jt) - 1) * hz )
    
    def neteff(self):
        i1 = np.log( 1 + (np.exp(self._i_0) - 1) * np.exp(self._g_0) )
        
        g_net = i1 / self._i_0
        g_eff = (i1 - self._i_0) / self._g_0
        
        return g_net, g_eff

    def plot(self, z, t, show=True, savepath=None):
        # labelsize = 18
        # fontsize = 24
        # font = {'family' : 'serif',
        #         'color'  : 'black',
        #         'weight' : 'normal',
        #         'size'   : fontsize,
        #         }
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')

        i = self.intensity(z, t)
        g = self.gain(z, t)

        fig, ax1 = plt.subplots(figsize=(8.0,6.0))
        color = '#1f77b4'
        ax1.plot(z, i, color=color)
        ax1.set_xlabel(r'$z$', fontdict=font)
        ax1.set_ylabel(r'$I(z)$', fontdict=font, color=color)
        ax1.tick_params(axis='x', labelsize=labelsize)
        ax1.tick_params(axis='y', labelsize=labelsize)
        ax1.set_xlim(z[0],z[-1])
        ax1.set_ylim(0.0, get_ylim()[1])
        ax2 = ax1.twinx()
        color = '#ff7f0e'
        ax2.plot(z, g, color=color)
        ax2.set_ylabel(r'$G(z)$', fontdict=font, color=color)
        ax2.tick_params(axis='y', labelsize=labelsize)
        ax2.set_ylim(0.0, get_ylim()[1])

        if show:
            plt.show()
        if savepath is None:
            plt.close()
        else:
            fig.savefig(savepath, bbox_inches='tight')
            plt.close()
            print("Saved {0}\n".format(savepath))

    def anim_pulse(self, z, t):
        z_mg, t_mg = np.meshgrid(z, t, sparse=False, indexing='ij')

        i = self.intensity(z_mg, t_mg)
        g = self.gain(z_mg, t_mg)
        g_net, g_eff = self.neteff()
        print("Net pulse energy gain: {:.{prec}}".format(g_net, prec=4))
        print("Energy extraction efficiency: {:.{prec}}".format(g_eff, prec=3))

        fig = plt.figure(figsize=(8,4))
        ax1 = plt.subplot(1,1,1)   

        txt_title = ax1.set_title('')
        ax1.set_xlim((z[0], z[-1]))            
        ax1.set_ylim((0, max(1.0, y_max(i/self._i_0))))
        ax1.set_xlabel('z')

        line1, = ax1.plot([], [], 'b', lw=2)
        line2, = ax1.plot([], [], 'r', lw=2)

        def drawframe(n):
            line1.set_data(z, g[:,n]/self._g_0)
            line2.set_data(z, i[:,n]/self._i_0)
            txt_title.set_text('Frame = {0:4d}'.format(n))
            return (line1, line2)

        # blit=True re-draws only the parts that have changed.
        anim = animation.FuncAnimation(fig, drawframe, frames=t_mg.shape[1], interval=20, blit=True)
        
        return anim, i, g

class LaserAmplifierPulsePropRect(LaserAmplifierPulseProp):
    def _it(self, t):
        '''Compute the current pulse intensity as a function of time (private).

        Parameters
        ----------
        t : numpy.ndarray.float64
            An array containing the z values of the
            beginning and end of the region; if z is an empty array,
            then assume that the region fills the resonator
    
        Returns
        --------
        kappa_qmn : numpy.ndarray.complex128
            Array of spatial coupling coefficients with the same
            shape as q, m, and n
        '''

        return (self._i_0 / self._tau_pulse) * ( np.heaviside(t, 0.0) * np.heaviside(self._tau_pulse - t, 0.0) )

    def _jt(self, t):
        '''Compute the current pulse fluence (the integrated pulse intensity)
           as a function of time.
        '''
        return self._i_0 * ( (t / self._tau_pulse) * np.heaviside(t, 0.0) * np.heaviside(self._tau_pulse - t, 0.0) + np.heaviside(t - self._tau_pulse, 1.0) )
    
    def _gz(self, z):
        '''Compute the current gain profile as a function of position within
           the amplifier (0 < z < 1)
        '''
        return self._g_0 * ( np.heaviside(z, 0.0) * np.heaviside(1 - z, 0.0) )

    def _hz(self, z):
        '''Compute the current exponential integrated gain as a function of
           position within the amplifier (0 < z < 1)
        '''
        return np.exp( self._g_0 * ( z * np.heaviside(z, 0.0) * np.heaviside(1 - z, 0.0) + np.heaviside(z - 1, 1.0) ) )

class LaserAmplifierPulsePropGauss(LaserAmplifierPulseProp):
    def _it(self, t):
        '''Compute the current pulse intensity as a function of time.
        '''
        return self._i_0 * (np.sqrt(2/np.pi)/self._tau_pulse) * np.exp(-2*(t/self._tau_pulse)**2)

    def _jt(self, t):
        '''Compute the current pulse fluence (the integrated pulse intensity)
           as a function of time.
        '''
        return (self._i_0 / 2) * ( 1 + erf(np.sqrt(2)*t/self._tau_pulse) )
    
    def _gz(self, z):
        '''Compute the current gain profile as a function of position within
           the amplifier (0 < z < 1)
        '''
        return self._g_0 * ( np.heaviside(z, 0.0) * np.heaviside(1 - z, 0.0) )

    def _hz(self, z):
        '''Compute the current exponential integrated gain as a function of
           position within the amplifier (0 < z < 1)
        '''
        return np.exp( self._g_0 * ( z * np.heaviside(z, 0.0) * np.heaviside(1 - z, 0.0) + np.heaviside(z - 1, 1.0) ) )




# class Config(Enum):
#     URL, SWL, SHB = range(3)

# class SMLaserParameters(object):
#     def __init__(self, r_1 = 0.0, r_2 = 0.0, tau_prp = 0.0, tau_par = 0.0,
#                  g_0 = 0.0, omega = 0.0, shift = 0.0):
#         self._r_1 = r_1
#         self._r_2 = r_2
#         self._r  = self._r_1 * self._r_2
        
#         self._tau_pho = -1.0/np.log(self._r)
#         self._tau_prp = tau_prp
#         self._tau_par = tau_par

#         self._g_0 = g_0
#         self._shift = shift
#         self._omega = omega
        
#     def __str__(self):
#         template = 'tau_pho/tau_grp:\t{:.{prec}}\ntau_par/tau_grp:\t{:.{prec}}\ntau_prp/tau_grp:\t{:.{prec}}\ntau_prp/tau_pho:\t{:.{prec}}\n\n' \
#             + 'g_0:\t{:.{prec}}\nshift:\t{:.{prec}}\nomega:\t{:.{prec}}\n'
#         param_str = template.format(self._tau_pho, self._tau_par, self._tau_prp, self._tau_prp/self._tau_pho,
#                                     self._g_0, self._shift, self._omega, prec = 3)
#         return param_str

# class SingleModeLaserModel(ODEIntegrator):
#     def __init__(self, config:Config, params:SMLaserParameters, p, s, j):
#         ODEIntegrator.__init__(self, params)
#         self._config = config

#         def _delta_kappa(self):
#             r_1 = self._params._r_1
#             r_2 = self._params._r_2
#             r = r_1 * r_2
#             beta = (r_1 + r_2) * (1 + np.sqrt(r)) / ((np.sqrt(r_1) + np.sqrt(r_2)) * np.sqrt(r))

#             norm = (np.sqrt(r) * np.log(1.0/(r))) / ( 2 * (np.sqrt(r_1) + np.sqrt(r_2)) * (1 - np.sqrt(r)) )
#             phi_0 = norm * ( self._params._g_0 / (1.0 + self._params._omega**2) + 0.5 * np.log(r) )
#             sqrt_phi = (3.0/(2.0*beta)) * (np.sqrt(1.0 + 8.0*beta*phi_0/9.0) - 1.0)
#             dkappa = sqrt_phi / (phi_0 - 0.5*sqrt_phi)
            
#             return dkappa

#         if (self._config == Config.URL):
#             self._kappa = 1.0
#         elif (self._config == Config.SWL):
#             self._kappa = 2.0
#         elif (self._config == Config.SHB):
#             self._kappa = 2.0 + self._delta_kappa()

#         self._p = p;
#         self._s = s;
#         self._j = j;

#     def _deriv(self, y, t):
#         dydt = np.zeros_like(y)
        
#         e = y[0] + 1j * y[1]
#         f = y[2] + 1j * y[3]
#         g = y[4]
        
#         dedt = (-(1.0 + self._s(t)) /(2.0 * self._params._tau_pho) + 1j * self._params._shift) * e + f
#         dfdt = -(1.0/self._params._tau_prp) * ( (1.0 - 1j * self._params._omega) * f - g * (e + self._j(t)) / 2.0 )
#         dgdt = -(1.0/self._params._tau_par) * ( g - self._params._g_0 * self._p(t) + 2.0 * self._kappa * np.real(np.conj(e) * f) )
        
#         dydt[0] = np.real(dedt)
#         dydt[1] = np.imag(dedt)
#         dydt[2] = np.real(dfdt)
#         dydt[3] = np.imag(dfdt)
#         dydt[4] = dgdt
        
#         self._calls += 1
    
#         return dydt

#     def _comp_efg(self, y):
#         e = y[:, 0] + 1j * y[:, 1]
#         f = y[:, 2] + 1j * y[:, 3]
#         g = y[:, 4]
        
#         return e, f, g

#     def _simplot(self, t, y, usetex, show, dirpath, format_string):
#         def y_max(y):
#             maxy = np.max(y)
#             shift = -int(np.log10(maxy)) + 1
            
#             return np.ceil(maxy*10**(shift))*10**(-shift)

#         def label_str(str, usetex):
#             if usetex:
#                 return r'$' + str + '$'
#             else:
#                 return str

#         plt.rc('text', usetex=usetex)
#         plt.rc('font', family='serif')
#         font = {'family' : 'serif',
#                 'color'  : 'black',
#                 'weight' : 'normal',
#                 'size'   : 18,
#                 }
        
#         show_plot = show[0]
#         if ( len(show) > 1 ):
#             show_cw = show[1]
#         else:
#             show_cw = True
        
#         e, f, g = self._comp_efg(y)

#         g_th = np.ones_like(t) * (1.0 + self._params._omega**2) / self._params._tau_pho
#         if (self._params._g_0 > g_th[0]):
#             e_cw = np.sqrt( (self._params._tau_pho / self._kappa) * (self._params._g_0 - g_th) )
#         else:
#             e_cw = np.zeros_like(t)
#         f_cw = 0.5 * g_th * e_cw / np.sqrt(1.0 + self._params._omega**2)
#         cos2_cw = np.ones_like(t) / (1.0 + self._params._omega**2)

#         fig, ax = plt.subplots(2, 2, figsize=(16, 12))
#         fig.subplots_adjust(hspace=0.25, wspace = 0.25)
#         for axis_row in ax:
#             for axis in axis_row:
#                 axis.tick_params(axis='both', labelsize=14)
#                 axis.grid(True)
#         ax[0,0].set_xlabel(label_str('t', usetex), fontdict=font)
#         ax[0,0].set_ylabel(label_str('|E(t)|, |F(t)|, G(t)', usetex), fontdict=font)
#         ax[0,0].set_xlim(t[0], t[-1])
#         ax[0,0].plot(t, np.abs(e), label = label_str('|E(t)|', usetex))
#         ax[0,0].plot(t, np.abs(f), label = label_str('|F(t)|', usetex))
#         ax[0,0].plot(t, g, label = label_str('G(t)', usetex))
#         ax[0,0].plot(t, g_th, '--', label = label_str('G_\mathrm{th}', usetex))
#         if show_cw:
#             ax[0,0].plot(t, e_cw, '--', label = label_str('E_\mathrm{cw}', usetex))
#             ax[0,0].plot(t, f_cw, '--', label = label_str('F_\mathrm{cw}', usetex))
#         ax[0,0].legend()
#         ax[0,1].set_xlabel(label_str('t', usetex), fontdict=font)
#         ax[0,1].set_ylabel(label_str('\phi_E(t), \phi_F(t)', usetex), fontdict=font)
#         ax[0,1].set_xlim(t[0], t[-1])
#         ax[0,1].plot(t, np.angle(e), label = label_str('\phi_E(t)', usetex))
#         ax[0,1].plot(t, np.angle(f), label = label_str('\phi_F(t)', usetex))
#         ax[0,1].legend()
#         ax[1,0].set_xlabel(label_str('t', usetex), fontdict=font)
#         ax[1,0].set_ylabel(label_str('p(t), s(t), j(t)', usetex), fontdict=font)
#         ax[1,0].set_xlim(t[0], t[-1])
#         ax[1,0].plot(t, self._p(t), label = label_str('p(t)', usetex))
#         ax[1,0].plot(t, self._s(t), label = label_str('s(t)', usetex))
#         ax[1,0].plot(t, self._j(t), label = label_str('j(t)', usetex))
#         ax[1,0].legend()
#         ax[1,1].set_xlabel(label_str('t', usetex), fontdict=font)
#         ax[1,1].set_ylabel(label_str('\cos^2 [\phi_E(t) - \phi_F(t)]', usetex), fontdict=font)
#         ax[1,1].set_xlim(t[0], t[-1])
#         ax[1,1].plot(t, np.cos(np.angle(e) - np.angle(f))**2, label = label_str('\cos^2 [\phi_E(t) - \phi_F(t)]', usetex))
#         ax[1,1].plot(t, cos2_cw, '--', label = label_str('1/(1 + \Omega^2)', usetex))
#         ax[1,1].legend()

#         if dirpath is not None:
#             savepath = self._savepath(dirpath, format_string)
#             fig.savefig(savepath, bbox_inches='tight')
#             print("\nSaved {0}".format(savepath))
#         if show_plot:
#             plt.show()
#         else:
#             plt.close()

# class SingleModeLaserModelREA(SingleModeLaserModel):
#     def _deriv(self, y, t):
#         dydt = np.zeros_like(y)
        
#         e = y[0] + 1j * y[1]
#         g = y[4]
#         f = g * (e + self._j(t)) / (2.0 * (1.0 - 1j * self._params._omega))
        
#         dedt = (-(1.0 + self._s(t)) /(2.0 * self._params._tau_pho) + 1j * self._params._shift) * e + f
#         dfdt = 0.0 + 0.0j
#         dgdt = -(1.0/self._params._tau_par) * ( g - self._params._g_0 * self._p(t) + 2.0 * self._kappa * np.real(np.conj(e) * f) )
        
#         dydt[0] = np.real(dedt)
#         dydt[1] = np.imag(dedt)
#         dydt[2] = np.real(dfdt)
#         dydt[3] = np.imag(dfdt)
#         dydt[4] = dgdt
        
#         self._calls += 1
    
#         return dydt

#     def _comp_efg(self, y):
#         e = y[:, 0] + 1j * y[:, 1]
#         g = y[:, 4]
#         f = g * e / (2.0 * (1.0 - 1j * self._params._omega))
        
#         return e, f, g
