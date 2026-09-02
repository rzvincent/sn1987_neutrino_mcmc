import definition as df
import numpy as np


coefficient = 3.89379e-10      # Unit transformation from eV^-2 to cm^2
Distance_SN1987 = df.Distance_SN1987        # Distance of SN 1987A in unit cm
T_nu = df.T_nu

# gamma_phi divided by m_phi
def omega(lambda_chi = 1):

    return lambda_chi**2/ (16*np.pi)

def alpha_m_phi(E_nu):
    return 2*E_nu*1e6 * T_nu


def alpha(E_nu, m_phi):
    return 2*E_nu*df.T_nu*1e6 / m_phi**2


def nr_gl(E_nu, log_m_phi, m_nu):
    """
    Compute the normalized rate using Gauss-Legendre integration.
    """

    m_phi = 10**log_m_phi

    x_0 = m_nu / T_nu

    beta = lambda x: np.sqrt(1- x_0**2/x**2)

    part1 = lambda x: x**2 * np.sqrt(x**2 - x_0**2) / (df.safe_exp(np.sqrt(x**2 - x_0**2)) + 1 ) 

    part2 = lambda c, x, E_nu: omega()*alpha(E_nu, m_phi)*(1-beta(x)*c)**2 / ((alpha(E_nu, m_phi)*x*(1-beta(x)*c) -1)**2 + omega()**2)

    integ_part = lambda c, x, E_nu: part1(x) * part2(c, x, E_nu)

    upper_x = 50/(x_0 + 2.5)

    ret = df.gl2_integrate_vec(integ_part, [(-1, 1), (x_0, x_0 + upper_x)], E_nu, n=128)

    return ret


def os_gl(E_nu, log_m_phi, log_lambda_nu, m_nu=df.m_nu):
    """
    Compute the overall scattering after traveling from SN1987 distance.
    Use the lightest neutrino mass eigenstate m_1 as default, but can be changed to m_2 or m_3.
    Inserted in signal rate.
    """

    E_space = np.linspace(4, df.E_max_2d, 15)   # this value should be chosen carefully, 15 is enough for the fitting, but 1000 is needed for likelihood-ratio calculation with the mock sample to generate a smooth curve (Fig.5 on arxiv version)

    lambda_nu = 10**log_lambda_nu

    m_phi = 10**log_m_phi

    part1 = T_nu**3 * lambda_nu**2 / (4*np.pi**2) / m_phi**2

    ratio =  -part1*nr_gl(E_space, log_m_phi, m_nu)* Distance_SN1987 * np.sqrt(1/coefficient)

    ret = df.func_build(E_nu, E_space, ratio)

    return np.clip(ret, None, 0)



# only for Fig.2 plotting, not for fitting
def sr_gl(E_nu, log_m_phi, x_0):
    """
    Compute the scattering rate without considering lambda_phi_nv using Gauss-Legendre integration.
    """
    m_phi = 10**log_m_phi

    beta = lambda x: np.sqrt(1- x_0**2/x**2)

    part1 = lambda x: x**2 * np.sqrt(x**2 - x_0**2) / (df.safe_exp(np.sqrt(x**2 - x_0**2)) + 1 ) 

    part2 = lambda c, x, E_nu: omega()*alpha(E_nu, m_phi)*(1-beta(x)*c)**2 / ((alpha(E_nu, m_phi)*x*(1-beta(x)*c) -1)**2 + omega()**2)

    integ_part = lambda c, x, E_nu: part1(x) * part2(c, x, E_nu)

    upper_x = 50/(x_0 + 2.5)

    ret = df.gl2_integrate_vec(integ_part, [(-1, 1), (x_0, x_0+upper_x)], E_nu, n=256)

    part3 = T_nu**3 / (4*np.pi**2) / m_phi**2

    ratio =  part3 * ret

    return ratio


def compute_row_sr(x_0, E_nu_0, log_m_phi):

    return [sr_gl(E_nu_0, log_m_phi, x_0) for log_m_phi in log_m_phi]


# def normalized_rate(E_nu, log_m_phi, x_0):      # In ev^-2  

#     m_phi = 10**log_m_phi

#     beta = lambda x: np.sqrt(1- x_0**2/x**2)

#     part1 = lambda x: x**2 / (df.safe_exp(x) + 1 )    # important in x \in [0., 20.]

#     part2 = lambda c, x: (1-beta(x)*c)**2 / np.sqrt(1+beta(x)**2- 2*beta(x)*c)

#     part3 = lambda c, x: omega()*alpha(E_nu, m_phi)*x / (omega()**2  + (alpha(E_nu, m_phi)*x*(1-beta(x)*c) -1)**2)

#     integ_part = lambda c, x: part1(x) * part2(c, x) * part3(c, x)

#     ret = df.gl2_integrate(integ_part, [(-1, 1), (x_0, x_0+20.)], n=1024)

#     return ret

# def compute_row(x_0, E_nu_0, log_m_phi):
#     """
#     Worker function that computes the inner loop (over log_m_phi_space) 
#     for a single x_0 value.
#     """
#     return [normalized_rate(E_nu_0, log_m_phi, x_0) for log_m_phi in log_m_phi]



if __name__ == '__main__':
    

    print(np.exp(os_gl(30, -2, -16, df.m_nu)))


    # E_space = np.linspace(4, df.E_max_2d, 1000)

    # plt.plot(E_space, d_phi_gl(E_space, 2, -6))
    # plt.plot(E_space, os_gl(E_space, 2, -6))
    # plt.show()