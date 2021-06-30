from scipy.optimize import leastsq, root
import numpy as np
from scipy import linalg as lin
from scipy import integrate
from matplotlib import pylab as plt

class LotkiVolterra:
    def __init__(self, a, b, c, d, e):
        self.a, self.b, self.c, self.d, self.e = a, b, c, d, e

    def dX_dt(self, X, t=0):
        """ Return the growth rate of fox and rabbit populations. """
        return np.array([ self.a*X[0] - self.b*X[0]*X[1] - self.e*X[0]**2 ,
                         -self.c*X[1] + self.d*self.b*X[0]*X[1] ])

    def d2X_dt2(self, X, t=0):
        """ Return the Jacobian matrix evaluated in X. """
        return np.array([[self.a -self.b*X[1] - 2*self.e*X[0],   -self.b*X[0]     ],
                         [self.b*self.d*X[1] ,   -self.c +self.b*self.d*X[0]] ])

    def analysis(self, t, X0):
        X_f0 = np.array([     0. ,  0.])
        X_f1 = np.array([ self.c/(self.d*self.b), self.a/self.b])
        if all(self.dX_dt(X_f0) == np.zeros(2) ) and all(self.dX_dt(X_f1) == np.zeros(2)):
            return None

        A_f0 = self.d2X_dt2(X_f0)
        A_f1 = self.d2X_dt2(X_f1)
        lambda1, lambda2 = np.linalg.eigvals(A_f1) # >>> (1.22474j, -1.22474j)
        # They are imaginary numbers. The fox and rabbit populations are periodic as follows from further
        # analysis. Their period is given by:
        print(lambda1, lambda2)
        T_f1 = 2*np.pi/abs(lambda1)                # >>> 5.130199

        X, infodict = integrate.odeint(self.dX_dt, X0, t, full_output=True)
        print(infodict['message'])
        return X

    def plot_x_t(self, name='rabbits_and_foxes_1.png'):
        t = np.linspace(0, 15, 1000)  # time
        X0 = np.array([10, 5])  # initials conditions: 10 rabbits and 5 foxes
        res = self.analysis(t, X0)
        if res is None:
            return None
        rabbits, foxes = res.T
        f1 = plt.figure()
        plt.plot(t, rabbits, 'r-', label='Rabbits')
        plt.plot(t, foxes  , 'b-', label='Foxes')
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('time')
        plt.ylabel('population')
        plt.title('Evolution of fox and rabbit populations')
        f1.savefig(name)
        plt.show()


def odu2(koeffs, func, L, bcl, bcr, N):
    """ u" + a u' + b u = f(x)
    b.c. : gamma * u' + beta * u = alpha
    koeffs = (a, b)
    func = f(x)
    L - длина отрезка по x (отрезок от 0 до L)
    bcl = lalpha, lbeta, lgamma
    bcr = ralpha, rbeta, rgamma
    N - число отрезков
    """
    a, b = koeffs
    lalpha, lbeta, lgamma = bcl
    ralpha, rbeta, rgamma = bcr
    h = L / N

    ''' u ->                       y[i],                *b
        u' ->   1 / 2h * (y[i+1]            - y[i-1])   *a
        u'' -> 1 / h**2 *(y[i+1] - 2y[i] + y[i-1])      *1

        '''
    ''' обозначения
    A0 - главная диагональ
    Аu1 - диагональ выше главной на 1
    Аd1 - диагональ ниже главной на 1'''
    A0 = np.ones(N + 1)  # y[i]
    Au1 = np.zeros(N + 1)  # y[i+1]
    Ad1 = np.zeros(N + 1)  # y[i-1]

    Au1[:] = a / (2 * h) + 1 / h ** 2
    A0[:] = b - 2 / h ** 2
    Ad1[:] = -a / (2 * h) + 1 / h ** 2
    F = np.fromfunction(func, (N + 1, 1))

    A0[0] = lbeta - lgamma / h
    Ad1[0] = 0
    Au1[0] = lgamma / h
    F[0] = lalpha

    A0[N] = rbeta + rgamma / h
    Au1[N] = 0
    Ad1[N] = -rgamma / h
    F[N] = ralpha

    Au1 = np.roll(Au1, 1)
    Ad1 = np.roll(Ad1, -1)
    A_band = np.concatenate((Au1, A0, Ad1)).reshape(3, N + 1)

    res = lin.solve_banded((1, 1), A_band, F)
    # print(res)
    return res


def analyse_ode2():
    L = np.pi
    N = 100
    y1 = odu2([0, 1], lambda x, y: -x * L / N, L, (0, 1, 0), (1, 0, 1), N)
    # print(y1)
    # u´´+u = -x
    f1 = plt.figure()
    plt.plot(np.linspace(0, L, N + 1), y1, '-')
    Ne = 10
    plt.plot(np.arange(0, L + L / Ne, L / Ne),
             np.fromfunction(lambda x, y: -2 * np.sin(x * L / Ne) - x * L / Ne,
                             (Ne + 1, 1)), '.')
    plt.show()


def odu4(koeffs, func, L, phi, psi, N):
    """ au(IV) + bu(III)+ cu" + du' + eu = f(x)
    г. у.: u(x=0) = phi[0], u(x=0) = phi[1]
           u'(x=0) = psi[0], u'(x=0) = psi[1]
    koeffs = (a, b, c, d, e)
    func = f(x)
    L - длина отрезка по x (отрезок от 0 до L)
    N - число отрезков
    """
    a, b, c, d, e = koeffs
    h = L / N

    ''' обозначения
    A0 - главная диагональ
    Аu1 - диагональ выше главной на 1
    Аu2 - диагональ выше главной на 2
    Аd1 - диагональ ниже главной на 1
    Аd2 - диагональ ниже главной на 2'''

    Au2 = np.zeros(N + 1)  # y[i+2]
    Au1 = np.zeros(N + 1)  # y[i+1]
    A0 = np.ones(N + 1)  # y[i]
    Ad1 = np.zeros(N + 1)  # y[i-1]
    Ad2 = np.zeros(N + 1)  # y[i-2]

    ''' u ->                       y[i],                *e
           u' ->   1 / 2h * (y[i+1]            - y[i-1])   *d
           u'' -> 1 / h**2 *(y[i+1] - 2y[i] + y[i-1])      *c
           u(III) -> 1/ 2h**3 *(-y[i-2]+2y[i-1]-2y[i+i]+y[i+2]) * b
           u(IV) -> 1 / h**4 *(y[i-2]-4y[i-1] +6y[i] -4y[i+1] + y[i+2]) * a
           '''
    F = np.fromfunction(func, (N + 1, 1))
    Au2[:] = a / (h ** 4) + b / (2 * (h ** 3))
    Au1[:] = d / (2 * h) + c / (h ** 2) - b / (h ** 3) - 4 * a / (h ** 4)
    A0[:] = e - 2 * c / (h ** 2) + 6 * a / (h ** 4)
    Ad1[:] = c / (h ** 2) - d / (2 * h) + b / (h ** 3) - 4 * a / (h ** 4)
    Ad2[:] = a / (h ** 4) - b / (2 * (h ** 3))

    Au2[0] = 0
    Au1[0] = 0
    A0[0] = 1
    Ad1[0] = 0
    Ad2[0] = 0
    F[0] = phi[0]

    Au2[1] = 0
    Au1[1] = 0
    A0[1] = 1 / h
    Ad1[1] = - 1 / h
    Ad2[1] = 0
    F[1] = psi[0]

    Au2[N - 1] = 0
    Au1[N - 1] = 1 / h
    A0[N - 1] = -1 / h
    Ad1[N - 1] = 0
    Ad2[N - 1] = 0
    F[N - 1] = psi[1]

    Au2[N] = 0
    Au1[N] = 0
    A0[N] = 1
    Ad1[N] = 0
    Ad2[N] = 0
    F[N] = phi[1]

    Au2 = np.roll(Au2, 2)
    Au1 = np.roll(Au1, 1)
    Ad1 = np.roll(Ad1, -1)
    Ad2 = np.roll(Ad2, -2)

    A_band = np.concatenate((Au2, Au1, A0, Ad1, Ad2)).reshape(5, N + 1)

    res = lin.solve_banded((2, 2), A_band, F)
    print(*res)
    return res


def analyse_ode4():
    L = 1
    N = 100
    y1 = odu4([1, 0, 0, 0, 1], lambda x, y: -x*L/N, L, (0, 0), (1, 1), N)

    f1 = plt.figure()
    plt.plot(np.linspace(0, L, N + 1), y1, '-')
    plt.show()


def find_root(func):
    z = root(func, 0)
    print('Корень уравнения:', *z.x)

    f1 = plt.figure()
    x = np.linspace(-5, 5, 101)
    y = func(x)
    plt.plot(z.x, 0, 'bo', x, y, '-g')
    plt.show()


def f(x):
    return [np.cos(x[0]) + x[1] - 1.5, 2*x[0] - np.sin(x[1]-0.5) - 1]


def find_root_sys():
    z = root(f, [0,0], method='lm')
    print('Корень системы:', z.x)
    x = np.linspace(0, 1, 100)
    res = [1.5 - np.cos(x), 0.5 + np.arcsin(2*x - 1)]
    f1 = plt.figure()
    plt.plot(z.x[0], z.x[1], 'bo', x, res[1], '-g', x, res[0], '-r')
    plt.show()


def least_sq(x, y):
    func = lambda params, x: params[0] * x ** 2 + params[1] * x + params[2]
    error = lambda params, x, y: func(params, x) - y

    pr_initial = (0.0, 0.0, 0.0)
    pr_final, success = leastsq(error, pr_initial[:], args=(x, y))

    print('Полная квадратичная невязка:', sum((y - func(pr_final, np.linspace(x.min(), x.max(), 5))) ** 2)**0.5)
    print('X = 21, Y = ', func(pr_final, 21))

    x1 = np.linspace(x.min(), x.max(), 50)
    y1 = func(pr_final, x1)
    plt.plot(x, y, 'bo', x1, y1, 'g-')
    plt.show()


def leastsq_radioact(T, N):
    func = lambda params, t: params[0] * np.exp(-params[1] * t)
    error = lambda params, x, y : func(params, x) - y

    pr_initial = (0, 0)
    pr_final, success = leastsq(error, pr_initial[:], args=(T, N))
    x = np.linspace(T.min(), T.max(), 50)
    y = func(pr_final, x)

    print('Константа распада:', pr_final[1])
    print('Период полураспада:', np.log(2) / pr_final[1])
    print('Полная квадратичная невязка:', sum((N - func(pr_final, np.linspace(T.min(), T.max(), 6))) ** 2)**0.5)

    f = plt.figure()
    plt.plot(x, y, 'g-', T, N, 'bo')
    plt.show()




def SIR(sir, t,  beta, gamma, N):
    # sir[0] - S, sir[1] - I, sir[2] - R
    dsdt = - (beta * sir[0] * sir[1])/N
    didt = (beta * sir[0] * sir[1])/N - gamma * sir[1]
    drdt = gamma * sir[1]
    dsirdt = [dsdt, didt, drdt]
    return dsirdt


if __name__ == "__main__":
    print('Задание 1')
    #analyse_ode2()
    analyse_ode4()

    print('\nЗадание 2.1')
    find_root( lambda x: x * x + 4 * np.cos(x) + 1)
    find_root_sys()

    print('\nЗадание 2.2.1')
    X = np.array([20, 22, 24, 26, 28])
    Y = np.array([40, 23.8, 14.1, 8.4, 5.0])
    least_sq(X, Y)

    print('\nЗадание 2.2.2')
    T = np.array([0, 0.664064, 1.328128 , 1.992192, 2.656256, 3.32032])
    N = np.array([1001429, 943124, 892257, 839982, 795584, 749496])
    leastsq_radioact(T, N)

    print('\nЗадание 3.2')
    a = 1.
    b = 0.1
    c = 1.5
    d = 0.75
    e = 0.1
    system = LotkiVolterra(a, b, c, d, e)
    system.plot_x_t('test.png')
    N = 1000
    S = N - 1
    I = 1
    R = 0
    beta = 0.3
    gamma = 0.2
    sir0 = (S, I, R)

    # time points
    t = np.linspace(0, 100)

    sir = integrate.odeint(SIR, sir0, t, args=(beta, gamma, N))

    plt.plot(t, sir[:, 0], label='S(t)')
    plt.plot(t, sir[:, 1], label='I(t)')
    plt.plot(t, sir[:, 2], label='R(t)')

    plt.legend()

    plt.xlabel('T')
    plt.ylabel('N')

    # use scientific notation
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.show()
    

