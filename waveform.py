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
        dLdt_LQG=-(16/5)*self.eta**2*self.r0*(1-e**2)**(3/2)/p**(9/2)*f_L_LQG
        dEdt=dEdt_GR+dEdt_LQG
        dLdt=dLdt_GR+dLdt_LQG
        return dEdt,dLdt
    def evolution_ode(self,t,y):
        p,e,_,_=y
        if e<0 or p<2*(1+e):
            return [0,0,0,0]
        dEdt,dLdt=self.get_fluxes(p,e)
        dEdt_norm=dEdt/self.eta
        dLdt_norm=dLdt/self.eta
        dpdt=-2*np.sqrt(p)*(p*dEdt_norm-np.sqrt(p/(1-e**2))*dLdt_norm)
        dedt=-(1-e**2)/e*(np.sqrt((1-e**2)/p)*dLdt_norm-(1-e**2)/np.sqrt(p)*dEdt_norm)
        omega_r,omega_phi=self.get_orbital_frequencies(p,e)
        dphi_r_dt=omega_r
        dphi_phi_dt=omega_phi
        return [dpdt,dedt,dphi_r_dt,dphi_phi_dt]
    
    def evolve_trajectory(self):
        y0=[self.p0,self.e0,0,0]
        t_span=[0,self.T_obs]
        def stop_condition(t,y):
            p,e,_,_=y
            return p-2*(1+e)
        
        stop_condition.terminal=True
        stop_condition.direction=-1
        sol=solve_ivp(
            self.evolution_ode,
            t_span,
            y0,
            method='RK45'
            dense_output=True,
            events=stop_condition,
            rtol=1e-9,atol=1e-10
        )
        self.evolution_results=sol
        return sol
    def generate_waveform(self,times,incl,phi0):
        sol=self.evolution_results.sol
        p_t,e_t,phi_r_t,phi_phi_t_rel=sol(times)
        phi_phi_t=phi_phi_t_rel+phi0
        amp_factor=2*self.mu/self.D_L
        h_plus=np.zeros_like(times)
        h_cross=np.zeros_like(times)
        n_max=30
        cos_i=np.cos(incl)
        sin_i=np.sin(incl)
        for n in range(1,n_max+1):
            arg=n*e_t
            J_n_minus_2=jv(n-2,arg)
            J_n_minus_1=jv(n-1,arg)
            J_n=jv(n,arg)
            J_n_plus_1=jv(n+1,arg)
            J_n_plus_2=jv(n+2,arg)
            g_n=n*phi_phi_t+phi_r_t
            A_n=-amp_factor*(n*omega_phi_t(p_t,e_t))**(2/3)*((1+cos_i**2)*(J_n_minus_2-2*e_t*J_n_minus_1+(2/n)*J_n+2*e_t*J_n_plus_1-J_n_plus_2)*np.cos(g_n)-sin_i**2*(J_n_minus_2-2*J_n+J_n_plus_2)*np.cos(g_n))
            B_n=-amp_factor*(n*omega_phi_t(p_t,e_t))**(2/3)*(-2*cos_i*(J_n_minus_2-2*e_t*J_n_minus_1+(2/n)*J_n-2*e_t*J_n_plus_1+J_n_plus_2)*np.sin(g_n))
        return h_plus,h_cross
    