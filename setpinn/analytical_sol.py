import numpy as np

def analytical_sol(eq_name, x, t):
    if eq_name == "wave":
        return u_ana1dwave(x, t)
    elif eq_name == "reaction":
        return u_ana1dreaction(x, t)
    elif eq_name == "convection":
        return u_anaconvection(x, t)
    elif eq_name == "plate":
        return partial_sum_rect_np(x, t)
    elif eq_name == "harmonic":
        return u_ana_plate_harmonic_scaled_np(x, t)
    else:
        raise ValueError("Invalid equation name")

def u_ana1dwave(x, t):
    """Analytical solution for the 1D wave"""
    return np.sin(np.pi * x) * np.cos(2 * np.pi * t) + 0.5 * np.sin(
        3 * np.pi * x
    ) * np.cos(6 * np.pi * t)

def h(x):
    """Helper function for the 1D reaction"""
    return np.exp(- (x - np.pi) ** 2 / (2 * (np.pi / 4) ** 2))

def u_ana1dreaction(x, t):
    """Analytical solution for the 1D reaction"""
    return h(x) * np.exp(5 * t) / (h(x) * np.exp(5 * t) + 1 - h(x))

def u_anaconvection(x, t):
    """Analytical solution for the heat convection"""
    return np.sin(x - 50 * t)

def partial_sum_rect_np(x, t, M=10, N=10, x0=0.25, x1=0.3, y0=0.7, y1=0.75, Q=20):
    """
    NumPy partial sum, interpreting (x,t) as (x,y).
    """
    return partial_sum_rect(
        x, t, M=M, N=N, x0=x0, x1=x1, y0=y0, y1=y1, Q=Q
    )
    
def coeff_mn_rect(m, n, x0, x1, y0, y1, Q=1.0):
    """
    Compute C_{m,n} for the rectangular patch load:
       -Delta(u)= Q in [x0,x1]x[y0,y1], else 0,
       with u=0 on boundary [0,1]^2.
    Closed-form integral for each sine term.
    """
    denom = (m*np.pi)**2 + (n*np.pi)**2

    # Integral over x0->x1 of sin(m pi x):
    # = (1/(m pi)) [cos(m pi x0)- cos(m pi x1)]
    Ix = (np.cos(m*np.pi*x0) - np.cos(m*np.pi*x1)) / (m*np.pi)

    # Integral over y0->y1 of sin(n pi y):
    Iy = (np.cos(n*np.pi*y0) - np.cos(n*np.pi*y1)) / (n*np.pi)

    # PDE sign: -Delta(u)= f => Delta(u)= -f => factor -1 => negative sign
    Cmn = - Q * (Ix * Iy) / denom
    return Cmn

def partial_sum_rect(x, y, M=10, N=10, x0=0.3, x1=0.4, y0=0.6, y1=0.65, Q=20):
    """
    Partial sum of the series up to (M,N).
    x,y can be scalar or arrays.
    """
    # Ensure arrays for vector ops
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    u_val = np.zeros_like(x, dtype=np.float64)

    for m in range(1, M+1):
        for n in range(1, N+1):
            Cmn = coeff_mn_rect(m, n, x0, x1, y0, y1, Q)
            u_val += Cmn * np.sin(m*np.pi*x) * np.sin(n*np.pi*y)

    return u_val

def u_ana_plate_harmonic_scaled_np(x, t, kx=5, ky=3, A=500.0):
    """
    NumPy version of the scaled plate solution,
    interpreting (x,t) as (x,y).
    """
    denom = (kx*np.pi)**2 + (ky*np.pi)**2
    return (A/denom)* np.sin(kx*np.pi*x)* np.sin(ky*np.pi*t)