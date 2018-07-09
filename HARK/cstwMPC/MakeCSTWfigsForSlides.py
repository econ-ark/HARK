'''
This module / script makes some fairly simple figures used in a version of the slides.
All Booleans at the top of SetupParamsCSTW should be set to False, as this module
imports cstwMPC; there's no need to actually do anything but load the model.
'''

from cstwMPC import *
import matplotlib.pyplot as plt

plot_range = (0.0,30.0)
points = 200
m = np.linspace(plot_range[0],plot_range[1],points)
InfiniteType(a_size=16)
InfiniteType.update()

thorn = 1.0025*0.99325/(1.01)
mTargFunc = lambda x : (1 - thorn)*x + thorn

mystr = lambda number : "{:.3f}".format(number)
mystrx = lambda number : "{:.0f}".format(number)

def epaKernel(X):
    K = 0.75*(1.0 - X**2.0)
    K[np.abs(X) > 1] = 0
    return K

def doCforBetaEquals(beta,m):
    InfiniteType(beta=beta);
    InfiniteType.solve();
    InfiniteType.unpack_cFunc()

    c = InfiniteType.cFunc[0](m)
    InfiniteType.beta = Params.beta_guess

    InfiniteType.simulateCSTWc()
    m_hist = InfiniteType.m_history
    m_temp = np.reshape(m_hist[100:Params.sim_periods,:],((Params.sim_periods-100)*Params.sim_pop_size,1))
    n = m_temp.size
    h = m[2] - m[0]
    m_dist = np.zeros(m.shape) + np.nan
    for j in range(m.size):
        x = (m_temp - m[j])/h
        m_dist[j] = np.sum(epaKernel(x))/(n*h)

    print('did beta= ' + str(beta))
    return c, m_dist

c_array = np.zeros((17,points)) + np.nan
pdf_array = np.zeros((17,points)) + np.nan
for b in range(17):
    beta = 0.978 + b*0.001
    c_array[b,], pdf_array[b,] = doCforBetaEquals(beta,m)

for b in range(17):
    beta = 0.978 + b*0.001
    highest = np.max(pdf_array[b,])
    scale = 1.5/highest
    scale = 4.0
    plt.ylim(0,2.5)
    plt.plot(m,scale*pdf_array[b,],'-c')
    plt.fill_between(m,np.zeros(m.shape),scale*pdf_array[b,],facecolor='c',alpha=0.5)
    plt.plot(m,mTargFunc(m),'-r')
    plt.plot(m,c_array[b,],'-k',linewidth=1.5)
    plt.text(10,2.2,r'$\beta=$' + str(beta),fontsize=20)
    plt.xlabel(r'Cash on hand $m_t$',fontsize=14)
    plt.ylabel(r'Consumption $c_t$',fontsize=14)
    plt.savefig('./Figures/mDistBeta0' + mystrx(1000*beta) + '.pdf')
    plt.show()


plt.plot(m,c_array[12,],'-k',linewidth=1.5)
plt.ylim(0,1.25)
plt.xlim(0,15)
plt.xlabel(r'Cash on hand $m_t$',fontsize=14)
plt.ylabel(r'Consumption $c_t$',fontsize=14)
plt.savefig('./Figures/ConFunc.pdf')
plt.plot(m,mTargFunc(m),'-r')
plt.plot(np.array([9.95,9.95]),np.array([0,1.5]),'--k')
plt.savefig('./Figures/mTargBase.pdf')
plt.fill_between(m,np.zeros(m.shape),scale*2*pdf_array[12,],facecolor='c',alpha=0.5)
plt.savefig('./Figures/mDistBase.pdf')
plt.show()

InfiniteType(beta=0.99);
InfiniteType.solve();
InfiniteType.unpack_cFunc()
m_new = np.linspace(0,15,points)
kappa_vec = InfiniteType.cFunc[0].derivative(m_new)

plt.plot(m_new,kappa_vec,'-k',linewidth=1.5)
plt.xlim(0,15)
plt.ylim(0,1.02)
plt.xlabel(r'Cash on hand $m_t$',fontsize=14)
plt.ylabel(r'Marginal consumption $\kappa_t$',fontsize=14)
plt.savefig('./Figures/kappaFuncBase.pdf')
plt.plot(np.array([9.95,9.95]),np.array([0,1.5]),'--k')
plt.fill_between(m,np.zeros(m.shape),scale*2*pdf_array[12,],facecolor='c',alpha=0.5)
plt.savefig('./Figures/mDistVsKappa.pdf')
plt.show()

plt.plot(m,mTargFunc(m),'-r')
plt.ylim(0,2.5)
plt.xlim(0,30)
for b in range(17):
    plt.plot(m,c_array[b,],'-k',linewidth=1.5)
    #idx = np.sum(c_array[b,] - mTargFunc(m) < 0)
    #mTarg = m[idx]
    #plt.plot(np.array([mTarg,mTarg]),np.array([0,2.5]),'--k')
plt.plot(m,mTargFunc(m),'-r')
plt.xlabel(r'Cash on hand $m_t$',fontsize=14)
plt.ylabel(r'Consumption $c_t$',fontsize=14)
plt.savefig('./Figures/ManycFuncs.pdf')
plt.show()

InfiniteType(beta=0.98);
InfiniteType.solve();
InfiniteType.unpack_cFunc()
m_new = np.linspace(0,15,points)
kappa_vec = InfiniteType.cFunc[0].derivative(m_new)

plt.plot(m_new,kappa_vec,'-k',linewidth=1.5)
plt.xlim(0,15)
plt.ylim(0,1.02)
plt.xlabel(r'Cash on hand $m_t$',fontsize=14)
plt.ylabel(r'Marginal consumption $\kappa_t$',fontsize=14)
plt.savefig('./Figures/kappaFuncLowBeta.pdf')
plt.fill_between(m,np.zeros(m.shape),scale*0.33*pdf_array[2,],facecolor='c',alpha=0.5)
plt.savefig('./Figures/mDistVsKappaLowBeta.pdf')
plt.show()
