#LQG waveform generator
import numpy as np
from scipy.integrate import quad,solve_ivp
from scipy.special import jv
#使用几何单位制
GM_SUN_S=4.925490947e-6
PC_M=3.08567758149e16
GPC_M=PC_M*1e9
class LQGWaveformGenerator:
    def __init__(self,M_sol,m_sol,r0,p0,e0,D_L,T_obs_yr=4):
        self.M_sol=M_sol
        self.m_sol=m_sol
        self.r0=r0
        self.p0=p0
        self.e0=e0
        self.D_L=D_L

        #单位转换
        self.M_sec=M_sol*GM_SUN_S
        self.m_sec=m_sol*GM_SUN_S
        self.mu_sec=self.m_sec
        self.mu=self.mu_sec/self.M_sec
        self.eta=self.mu
        self.D_L=(D_L*GPC_M)/(self.M_sec*299792458)
        self.T_obs=T_obs_yr*31536000/self.M_sec

    def get_orbital_frequencies(self,p,e):
        omega_phi_sq=(1/p**3)*(1+self.r0/(2*p))*(3-e**2)
        omega_phi=np.sqrt(omega_phi_sq)
        delta_r_phi=2*np.pi*(1+(3/p)+(self.r0/(4*p**2))*(15-e**2))
        omega_r=2*np.pi*omega_phi/delta_r_phi
        return omega_r,omega_phi
    
    def get_fluxes(self,p,e):
        f_E_GR=(1+(73/24)*e**2*(37/96)*e**4)
        f_L_GR=(1+(7/8)*e**2)

        dEdt_GR=-(32/5)*self.eta**2*(1-e**2)**(3/2)/p**5*f_E_GR
        dLdt_GR=-(32/5)*self.eta**2*(1-e**2)**(3/2)/p**(7/2)*f_L_GR

        f_E_LQG=(1+(181/24)*e**2+(329/96)*e**4)
        f_L_LQG=(1+(25/8)*e**2)

        dEdt_LQG=-(16/5)*self.eta**2*self.r0*(1-e**2)**(3/2)/p**6*f_E_LQG
        dLdt_LQG=-(16/5)*self.eta**2*self.r0*(1-e**2)**(3/2)/p**(9/2)*f_L_GR
        dEdt=dEdt_GR+dEdt_LQG
        dLdt=dLdt_GR+dLdt_LQG
        return dEdt,dLdt
    