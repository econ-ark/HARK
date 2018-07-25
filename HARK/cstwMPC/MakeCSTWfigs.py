'''
This module makes some figures for cstwMPC.  It requires that quite a few specifications
of the model have been estimated, with the results stored in ./Results.
'''

from builtins import range
import matplotlib.pyplot as plt
import csv
import numpy as np
import os

# Save the current file's directory location for writing output:
my_file_path = os.path.dirname(os.path.abspath(__file__))


f = open(my_file_path  + '/Results/LCbetaPointNetWorthLorenzFig.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
lorenz_percentiles = []
scf_lorenz = []
beta_point_lorenz = []
for j in range(len(raw_data)):
    lorenz_percentiles.append(float(raw_data[j][0]))
    scf_lorenz.append(float(raw_data[j][1]))
    beta_point_lorenz.append(float(raw_data[j][2]))
f.close()
lorenz_percentiles = np.array(lorenz_percentiles)
scf_lorenz = np.array(scf_lorenz)
beta_point_lorenz = np.array(beta_point_lorenz)

f = open(my_file_path  + '/Results/LCbetaDistNetWorthLorenzFig.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
beta_dist_lorenz = []
for j in range(len(raw_data)):
    beta_dist_lorenz.append(float(raw_data[j][2]))
f.close()
beta_dist_lorenz = np.array(beta_dist_lorenz)

f = open(my_file_path  + '/Results/LCbetaPointNetWorthMPCfig.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
mpc_percentiles = []
mpc_beta_point = []
for j in range(len(raw_data)):
    mpc_percentiles.append(float(raw_data[j][0]))
    mpc_beta_point.append(float(raw_data[j][1]))
f.close()
mpc_percentiles = np.asarray(mpc_percentiles)
mpc_beta_point = np.asarray(mpc_beta_point)

f = open(my_file_path  + '/Results/LCbetaDistNetWorthMPCfig.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
mpc_beta_dist = []
for j in range(len(raw_data)):
    mpc_beta_dist.append(float(raw_data[j][1]))
f.close()
mpc_beta_dist = np.asarray(mpc_beta_dist)

f = open(my_file_path  + '/Results/LCbetaDistLiquidMPCfig.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
mpc_beta_dist_liquid = []
for j in range(len(raw_data)):
    mpc_beta_dist_liquid.append(float(raw_data[j][1]))
f.close()
mpc_beta_dist_liquid = np.asarray(mpc_beta_dist_liquid)

f = open(my_file_path  + '/Results/LCbetaDistNetWorthKappaByAge.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
kappa_mean_age = []
kappa_lo_beta_age = []
kappa_hi_beta_age = []
for j in range(len(raw_data)):
    kappa_mean_age.append(float(raw_data[j][0]))
    kappa_lo_beta_age.append(float(raw_data[j][1]))
    kappa_hi_beta_age.append(float(raw_data[j][2]))
kappa_mean_age = np.array(kappa_mean_age)
kappa_lo_beta_age = np.array(kappa_lo_beta_age)
kappa_hi_beta_age = np.array(kappa_hi_beta_age)
age_list = np.array(list(range(len(kappa_mean_age))),dtype=float)*0.25 + 24.0
f.close()

f = open(my_file_path  + '/Results/LC_KYbyBeta.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
KY_by_beta_lifecycle = []
beta_list = []
for j in range(len(raw_data)):
    beta_list.append(float(raw_data[j][0]))
    KY_by_beta_lifecycle.append(float(raw_data[j][1]))
beta_list = np.array(beta_list)
KY_by_beta_lifecycle = np.array(KY_by_beta_lifecycle)
f.close()

f = open(my_file_path  + '/Results/IH_KYbyBeta.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
KY_by_beta_infinite = []
for j in range(len(raw_data)):
    KY_by_beta_infinite.append(float(raw_data[j][1]))
KY_by_beta_infinite = np.array(KY_by_beta_infinite)
f.close()


plt.plot(100*lorenz_percentiles,beta_point_lorenz,'-.k',linewidth=1.5)
plt.plot(100*lorenz_percentiles,beta_dist_lorenz,'--k',linewidth=1.5)
plt.plot(100*lorenz_percentiles,scf_lorenz,'-k',linewidth=1.5)
plt.xlabel('Wealth percentile',fontsize=14)
plt.ylabel('Cumulative wealth ownership',fontsize=14)
plt.title('Lorenz Curve Matching, Lifecycle Model',fontsize=16)
plt.legend((r'$\beta$-point',r'$\beta$-dist','SCF data'),loc=2,fontsize=12)
plt.ylim(-0.01,1)
plt.savefig(my_file_path  + '/Figures/LorenzLifecycle.pdf')
plt.show()

plt.plot(mpc_beta_point,mpc_percentiles,'-.k',linewidth=1.5)
plt.plot(mpc_beta_dist,mpc_percentiles,'--k',linewidth=1.5)
plt.plot(mpc_beta_dist_liquid,mpc_percentiles,'-.k',linewidth=1.5)
plt.xlabel('Marginal propensity to consume',fontsize=14)
plt.ylabel('Cumulative probability',fontsize=14)
plt.title('CDF of the MPC, Lifecycle Model',fontsize=16)
plt.legend((r'$\beta$-point NW',r'$\beta$-dist NW',r'$\beta$-dist LA'),loc=0,fontsize=12)
plt.savefig(my_file_path  + '/Figures/MPCdistLifecycle.pdf')
plt.show()

plt.plot(age_list,kappa_mean_age,'-k',linewidth=1.5)
plt.plot(age_list,kappa_lo_beta_age,'--k',linewidth=1.5)
plt.plot(age_list,kappa_hi_beta_age,'-.k',linewidth=1.5)
plt.legend(('Population average','Most impatient','Most patient'),loc=2,fontsize=12)
plt.xlabel('Age',fontsize=14)
plt.ylabel('Average MPC',fontsize=14)
plt.title('Marginal Propensity to Consume by Age',fontsize=16)
plt.xlim(24,100)
plt.ylim(0,1)
plt.savefig(my_file_path  + '/Figures/MPCbyAge.pdf')
plt.show()

plt.plot(beta_list,KY_by_beta_infinite,'-k',linewidth=1.5)
plt.plot(beta_list,KY_by_beta_lifecycle,'--k',linewidth=1.5)
plt.plot([0.95,1.01],[10.26,10.26],'--k',linewidth=0.75)
plt.text(0.96,12,'U.S. K/Y ratio')
plt.legend(('Perpetual youth','Lifecycle'),loc=2,fontsize=12)
plt.xlabel(r'Discount factor $\beta$',fontsize=14)
plt.ylabel('Capital to output ratio',fontsize=14)
plt.title('K/Y Ratio by Discount Factor',fontsize=16)
plt.ylim(0,100)
plt.savefig(my_file_path  + '/Figures/KYratioByBeta.pdf')
plt.show()


f = open(my_file_path  + '/Results/IHbetaPointNetWorthLorenzFig.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
lorenz_percentiles = []
scf_lorenz = []
beta_point_lorenz = []
for j in range(len(raw_data)):
    lorenz_percentiles.append(float(raw_data[j][0]))
    scf_lorenz.append(float(raw_data[j][1]))
    beta_point_lorenz.append(float(raw_data[j][2]))
f.close()
lorenz_percentiles = np.array(lorenz_percentiles)
scf_lorenz = np.array(scf_lorenz)
beta_point_lorenz = np.array(beta_point_lorenz)

f = open(my_file_path  + '/Results/IHbetaDistNetWorthLorenzFig.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
beta_dist_lorenz = []
for j in range(len(raw_data)):
    beta_dist_lorenz.append(float(raw_data[j][2]))
f.close()
beta_dist_lorenz = np.array(beta_dist_lorenz)


f = open(my_file_path  + '/Results/IHbetaPointLiquidLorenzFig.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
beta_point_lorenz_liquid = []
for j in range(len(raw_data)):
    beta_point_lorenz_liquid.append(float(raw_data[j][2]))
f.close()
beta_point_lorenz_liquid = np.array(beta_point_lorenz_liquid)

f = open(my_file_path  + '/Results/IHbetaDistLiquidLorenzFig.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
beta_dist_lorenz_liquid = []
for j in range(len(raw_data)):
    beta_dist_lorenz_liquid.append(float(raw_data[j][2]))
f.close()
beta_dist_lorenz_liquid = np.array(beta_dist_lorenz_liquid)

f = open(my_file_path  + '/Results/IHbetaPointNetWorthMPCfig.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
mpc_percentiles = []
mpc_beta_point = []
for j in range(len(raw_data)):
    mpc_percentiles.append(float(raw_data[j][0]))
    mpc_beta_point.append(float(raw_data[j][1]))
f.close()
mpc_percentiles = np.asarray(mpc_percentiles)
mpc_beta_point = np.asarray(mpc_beta_point)

f = open(my_file_path  + '/Results/IHbetaDistNetWorthMPCfig.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
mpc_beta_dist = []
for j in range(len(raw_data)):
    mpc_beta_dist.append(float(raw_data[j][1]))
f.close()
mpc_beta_dist = np.asarray(mpc_beta_dist)

f = open(my_file_path  + '/Results/IHbetaDistLiquidMPCfig.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
mpc_beta_dist_liquid = []
for j in range(len(raw_data)):
    mpc_beta_dist_liquid.append(float(raw_data[j][1]))
f.close()
mpc_beta_dist_liquid = np.asarray(mpc_beta_dist_liquid)


plt.plot(100*lorenz_percentiles,beta_point_lorenz,'-.k',linewidth=1.5)
plt.plot(100*lorenz_percentiles,scf_lorenz,'-k',linewidth=1.5)
plt.xlabel('Wealth percentile',fontsize=14)
plt.ylabel('Cumulative wealth ownership',fontsize=14)
plt.title('Lorenz Curve Matching, Perpetual Youth Model',fontsize=16)
plt.legend((r'$\beta$-point','SCF data'),loc=2,fontsize=12)
plt.ylim(-0.01,1)
plt.savefig(my_file_path  + '/Figures/LorenzInfiniteBP.pdf')
plt.show()

plt.plot(100*lorenz_percentiles,beta_point_lorenz,'-.k',linewidth=1.5)
plt.plot(100*lorenz_percentiles,beta_dist_lorenz,'--k',linewidth=1.5)
plt.plot(100*lorenz_percentiles,scf_lorenz,'-k',linewidth=1.5)
plt.xlabel('Wealth percentile',fontsize=14)
plt.ylabel('Cumulative wealth ownership',fontsize=14)
plt.title('Lorenz Curve Matching, Perpetual Youth Model',fontsize=16)
plt.legend((r'$\beta$-point',r'$\beta$-dist','SCF data'),loc=2,fontsize=12)
plt.ylim(-0.01,1)
plt.savefig(my_file_path  + '/Figures/LorenzInfinite.pdf')
plt.show()

plt.plot(100*lorenz_percentiles,beta_point_lorenz_liquid,'-.k',linewidth=1.5)
plt.plot(100*lorenz_percentiles,beta_dist_lorenz_liquid,'--k',linewidth=1.5)
plt.plot(np.array([20,40,60,80]),np.array([0.0, 0.004, 0.025,0.117]),'.r',markersize=10)
plt.xlabel('Wealth percentile',fontsize=14)
plt.ylabel('Cumulative wealth ownership',fontsize=14)
plt.title('Lorenz Curve Matching, Perpetual Youth (Liquid Assets)',fontsize=16)
plt.legend((r'$\beta$-point',r'$\beta$-dist','SCF targets'),loc=2,fontsize=12)
plt.ylim(-0.01,1)
plt.savefig(my_file_path  + '/Figures/LorenzLiquid.pdf')
plt.show()

plt.plot(mpc_beta_point,mpc_percentiles,'-.k',linewidth=1.5)
plt.plot(mpc_beta_dist,mpc_percentiles,'--k',linewidth=1.5)
plt.plot(mpc_beta_dist_liquid,mpc_percentiles,'-.k',linewidth=1.5)
plt.xlabel('Marginal propensity to consume',fontsize=14)
plt.ylabel('Cumulative probability',fontsize=14)
plt.title('CDF of the MPC, Perpetual Youth Model',fontsize=16)
plt.legend((r'$\beta$-point NW',r'$\beta$-dist NW',r'$\beta$-dist LA'),loc=0,fontsize=12)
plt.savefig(my_file_path  + '/Figures/MPCdistInfinite.pdf')
plt.show()




f = open(my_file_path  + '/Results/SensitivityRho.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
rho_sensitivity = np.array(raw_data)
f.close()

f = open(my_file_path  + '/Results/SensitivityXiSigma.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
xi_sigma_sensitivity = np.array(raw_data)
f.close()

f = open(my_file_path  + '/Results/SensitivityPsiSigma.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
psi_sigma_sensitivity = np.array(raw_data)
f.close()

f = open(my_file_path  + '/Results/SensitivityMu.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
mu_sensitivity = np.array(raw_data)
f.close()

f = open(my_file_path  + '/Results/SensitivityUrate.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
urate_sensitivity = np.array(raw_data)
f.close()

f = open(my_file_path  + '/Results/SensitivityMortality.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
mortality_sensitivity = np.array(raw_data)
f.close()

f = open(my_file_path  + '/Results/SensitivityG.txt','r')
my_reader = csv.reader(f,delimiter='\t')
raw_data = list(my_reader)
g_sensitivity = np.array(raw_data)
f.close()


#plt.subplots(2,3,sharey=True)
kappa = 0.242
plt.subplot(2,3,1,xticks=[0.5,2.3,4],ylim=(0.21,0.36))
plt.plot(rho_sensitivity[:,0],rho_sensitivity[:,1],'-k',linewidth=1)
plt.plot(1,kappa,'.k',markersize=8)
plt.yticks([0.22,0.26,0.30,0.34])
plt.text(0.7,0.215,r'Risk aversion $\rho$',fontsize=11)
plt.ylabel('Aggregate MPC',fontsize=11)

plt.subplot(2,3,2,xticks=[0.0,0.4,0.8],xlim=(0,0.8),ylim=(0.21,0.36))
plt.plot(xi_sigma_sensitivity[:,0],xi_sigma_sensitivity[:,1],'-k',linewidth=1)
plt.plot(0.2,kappa,'.k',markersize=8)
plt.yticks([0.22,0.26,0.30,0.34],[])
plt.text(0.045,0.215,r'Transitory std $\sigma_\theta$',fontsize=11)
plt.title('Sensitivity Analysis: Perpetual Youth',fontsize=14)

plt.subplot(2,3,3,xticks=[0.04,0.06,0.08],xlim=(0.04,0.08),ylim=(0.21,0.36))
plt.plot(psi_sigma_sensitivity[:,0],psi_sigma_sensitivity[:,1],'-k',linewidth=1)
plt.plot(0.0603,kappa,'.k',markersize=8)
plt.yticks([0.22,0.26,0.30,0.34],[])
plt.text(0.041,0.34,r'Permanent std $\sigma_\psi$',fontsize=11)

plt.subplot(2,3,4,xticks=[0.0,0.4,0.8],ylim=(0.21,0.36))
plt.plot(mu_sensitivity[:,0],mu_sensitivity[:,1],'-k',linewidth=1)
plt.plot(0.15,kappa,'.k',markersize=8)
plt.yticks([0.22,0.26,0.30,0.34])
plt.text(0.05,0.34,'Unemployment',fontsize=11)
plt.text(0.22,0.32,r'benefit $\mu$',fontsize=11)
plt.ylabel('Aggregate MPC',fontsize=11)

plt.subplot(2,3,5,xticks=[0.02,0.07,0.12],ylim=(0.21,0.36))
plt.plot(urate_sensitivity[:,0],urate_sensitivity[:,1],'-k',linewidth=1)
plt.plot(0.07,kappa,'.k',markersize=8)
plt.yticks([0.22,0.26,0.30,0.34],[])
plt.text(0.03,0.34,'Unemployment',fontsize=11)
plt.text(0.055,0.32,r'rate $\mho$',fontsize=11)

'''
plt.subplot(2,3,6,xticks=[0.004,0.008,0.012],xlim=(0.003,0.0125),ylim=(0.21,0.36))
plt.plot(mortality_sensitivity[:,0],mortality_sensitivity[:,1],'-k',linewidth=1)
plt.plot(0.00625,kappa,'.k',markersize=8)
plt.yticks([0.22,0.26,0.30,0.34],[])
plt.text(0.0037,0.34,r'Mortality rate $\mathsf{D}$',fontsize=11)
'''
plt.subplot(2,3,6,xlim=(0.0,0.04),ylim=(0.21,0.36))
plt.plot(g_sensitivity[:,0],g_sensitivity[:,1],'-k',linewidth=1)
plt.plot(0.01,kappa,'.k',markersize=8)
plt.xticks([0.0,0.02,0.04],['0','0.02','0.04'])
plt.yticks([0.22,0.26,0.30,0.34],[])
plt.text(0.008,0.34,r'Aggregate',fontsize=11)
plt.text(0.005,0.32,r'growth rate $g$',fontsize=11)

plt.savefig(my_file_path  + '/Figures/KappaSensitivity.pdf')
plt.show()



#plt.subplots(2,3,sharey=True)
beta = 0.9877
plt.subplot(2,3,1,xticks=[0.5,2.3,4],ylim=(0.95,1.0))
plt.plot(rho_sensitivity[:,0],rho_sensitivity[:,2],'-k',linewidth=1)
plt.plot(1,beta,'.k',markersize=8)
plt.yticks([0.95,0.96,0.97,0.98,0.99,1.0])
plt.text(0.7,0.955,r'Risk aversion $\rho$',fontsize=11)
plt.ylabel(r'Estimated $\grave{\beta}$',fontsize=11)

plt.subplot(2,3,2,xticks=[0.0,0.4,0.8],xlim=(0,0.8),ylim=(0.95,1.0))
plt.plot(xi_sigma_sensitivity[:,0],xi_sigma_sensitivity[:,2],'-k',linewidth=1)
plt.plot(0.2,beta,'.k',markersize=8)
plt.yticks([0.95,0.96,0.97,0.98,0.99,1.0],[])
plt.text(0.045,0.955,r'Transitory std $\sigma_\theta$',fontsize=11)
plt.title('Sensitivity Analysis: Perpetual Youth',fontsize=14)

plt.subplot(2,3,3,xticks=[0.04,0.06,0.08],xlim=(0.04,0.08),ylim=(0.97,1.0))
plt.plot(psi_sigma_sensitivity[:,0],psi_sigma_sensitivity[:,2],'-k',linewidth=1)
plt.plot(0.0603,beta,'.k',markersize=8)
plt.yticks([0.95,0.96,0.97,0.98,0.99,1.0],[])
plt.text(0.041,0.955,r'Permanent std $\sigma_\psi$',fontsize=11)

plt.subplot(2,3,4,xticks=[0.0,0.4,0.8],ylim=(0.95,1.0))
plt.plot(mu_sensitivity[:,0],mu_sensitivity[:,2],'-k',linewidth=1)
plt.plot(0.15,beta,'.k',markersize=8)
plt.yticks([0.95,0.96,0.97,0.98,0.99,1.0])
plt.text(0.05,0.9625,'Unemployment',fontsize=11)
plt.text(0.22,0.955,r'benefit $\mu$',fontsize=11)
plt.ylabel(r'Estimated $\grave{\beta}$',fontsize=11)

plt.subplot(2,3,5,xticks=[0.02,0.07,0.12],ylim=(0.95,1.0))
plt.plot(urate_sensitivity[:,0],urate_sensitivity[:,2],'-k',linewidth=1)
plt.plot(0.07,beta,'.k',markersize=8)
plt.yticks([0.95,0.96,0.97,0.98,0.99,1.0],[])
plt.text(0.03,0.9625,'Unemployment',fontsize=11)
plt.text(0.055,0.955,r'rate $\mho$',fontsize=11)

'''
plt.subplot(2,3,6,xticks=[0.004,0.008,0.012],xlim=(0.003,0.0125),ylim=(0.95,1.0))
plt.plot(mortality_sensitivity[:,0],mortality_sensitivity[:,2],'-k',linewidth=1)
plt.plot(0.00625,beta,'.k',markersize=8)
plt.yticks([0.95,0.96,0.97,0.98,0.99,1.0],[])
plt.text(0.0037,0.955,r'Mortality rate $\mathsf{D}$',fontsize=11)
'''
plt.subplot(2,3,6,xlim=(0.0,0.04),ylim=(0.95,1.0))
plt.plot(g_sensitivity[:,0],g_sensitivity[:,2],'-k',linewidth=1)
plt.plot(0.01,beta,'.k',markersize=8)
plt.xticks([0.0,0.02,0.04],['0','0.02','0.04'])
plt.yticks([0.95,0.96,0.97,0.98,0.99,1.0],[])
plt.text(0.008,0.9625,r'Aggregate',fontsize=11)
plt.text(0.005,0.955,r'growth rate $g$',fontsize=11)

plt.savefig(my_file_path  + '/Figures/BetaSensitivity.pdf')
plt.show()



#plt.subplots(2,3,sharey=True)
nabla = 0.00736
plt.subplot(2,3,1,xticks=[0.5,2.3,4],ylim=(0.000,0.055))
plt.plot(rho_sensitivity[:,0],rho_sensitivity[:,3],'-k',linewidth=1)
plt.plot(1,nabla,'.k',markersize=8)
plt.yticks([0,0.01,0.02,0.03,0.04,0.05])
plt.text(0.63,0.0475,r'Risk aversion $\rho$',fontsize=11)
plt.ylabel(r'Estimated $\nabla$',fontsize=11)

plt.subplot(2,3,2,xticks=[0.0,0.4,0.8],xlim=(0,0.8),ylim=(0.000,0.055))
plt.plot(xi_sigma_sensitivity[:,0],xi_sigma_sensitivity[:,3],'-k',linewidth=1)
plt.plot(0.2,nabla,'.k',markersize=8)
plt.yticks([0,0.01,0.02,0.03,0.04,0.05],[])
plt.text(0.045,0.0475,r'Transitory std $\sigma_\theta$',fontsize=11)
plt.title('Sensitivity Analysis: Perpetual Youth',fontsize=14)

plt.subplot(2,3,3,xticks=[0.04,0.06,0.08],xlim=(0.04,0.08),ylim=(0.00,0.055))
plt.plot(psi_sigma_sensitivity[:,0],psi_sigma_sensitivity[:,3],'-k',linewidth=1)
plt.plot(0.0603,nabla,'.k',markersize=8)
plt.yticks([0,0.01,0.02,0.03,0.04,0.05],[])
plt.text(0.041,0.0475,r'Permanent std $\sigma_\psi$',fontsize=11)

plt.subplot(2,3,4,xticks=[0.0,0.4,0.8],ylim=(0.000,0.055))
plt.plot(mu_sensitivity[:,0],mu_sensitivity[:,3],'-k',linewidth=1)
plt.plot(0.15,nabla,'.k',markersize=8)
plt.yticks([0,0.01,0.02,0.03,0.04,0.05])
plt.text(0.05,0.0475,'Unemployment',fontsize=11)
plt.text(0.22,0.040,r'benefit $\mu$',fontsize=11)
plt.ylabel(r'Estimated $\nabla$',fontsize=11)

plt.subplot(2,3,5,xticks=[0.02,0.07,0.12],ylim=(0.000,0.055))
plt.plot(urate_sensitivity[:,0],urate_sensitivity[:,3],'-k',linewidth=1)
plt.plot(0.07,nabla,'.k',markersize=8)
plt.yticks([0,0.01,0.02,0.03,0.04,0.05],[])
plt.text(0.03,0.0475,'Unemployment',fontsize=11)
plt.text(0.055,0.04,r'rate $\mho$',fontsize=11)

'''
plt.subplot(2,3,6,xticks=[0.004,0.008,0.012],xlim=(0.003,0.0125),ylim=(0.000,0.055))
plt.plot(mortality_sensitivity[:,0],mortality_sensitivity[:,3],'-k',linewidth=1)
plt.plot(0.00625,nabla,'.k',markersize=8)
plt.yticks([0,0.01,0.02,0.03,0.04,0.05],[])
plt.text(0.0037,0.0475,r'Mortality rate $\mathsf{D}$',fontsize=11)
'''

plt.subplot(2,3,6,xlim=(0.0,0.04),ylim=(0.000,0.055))
plt.plot(g_sensitivity[:,0],g_sensitivity[:,3],'-k',linewidth=1)
plt.plot(0.01,nabla,'.k',markersize=8)
plt.xticks([0.0,0.02,0.04],['0','0.02','0.04'])
plt.yticks([0,0.01,0.02,0.03,0.04,0.05],[])
plt.text(0.008,0.0475,r'Aggregate',fontsize=11)
plt.text(0.005,0.04,r'growth rate $g$',fontsize=11)

plt.savefig(my_file_path  + '/Figures/NablaSensitivity.pdf')
plt.show()



#plt.subplots(2,3,sharey=True)
fit = 4.593
plt.subplot(2,3,1,xticks=[0.5,2.3,4],ylim=(0,10))
plt.plot(rho_sensitivity[:,0],rho_sensitivity[:,4],'-k',linewidth=1)
plt.plot(1,fit,'.k',markersize=8)
plt.yticks([1,3,5,7,9])
plt.text(0.7,0.5,r'Risk aversion $\rho$',fontsize=11)
plt.ylabel('Lorenz distance',fontsize=11)

plt.subplot(2,3,2,xticks=[0.0,0.4,0.8],xlim=(0,0.8),ylim=(0,10))
plt.plot(xi_sigma_sensitivity[:,0],xi_sigma_sensitivity[:,4],'-k',linewidth=1)
plt.plot(0.2,fit,'.k',markersize=8)
plt.yticks([1,3,5,7,9],[])
plt.text(0.05,0.5,r'Transitory std $\sigma_\theta$',fontsize=11)
plt.title('Sensitivity Analysis: Perpetual Youth',fontsize=14)

plt.subplot(2,3,3,xticks=[0.04,0.06,0.08],xlim=(0.04,0.08),ylim=(0,10))
plt.plot(psi_sigma_sensitivity[:,0],psi_sigma_sensitivity[:,4],'-k',linewidth=1)
plt.plot(0.0603,fit,'.k',markersize=8)
plt.yticks([1,3,5,7,9],[])
plt.text(0.041,0.5,r'Permanent std $\sigma_\psi$',fontsize=11)

plt.subplot(2,3,4,xticks=[0.0,0.4,0.8],ylim=(0,10))
plt.plot(mu_sensitivity[:,0],mu_sensitivity[:,4],'-k',linewidth=1)
plt.plot(0.15,fit,'.k',markersize=8)
plt.yticks([1,3,5,7,9])
plt.text(0.05,8.5,'Unemployment',fontsize=11)
plt.text(0.22,7.25,r'benefit $\mu$',fontsize=11)
plt.ylabel('Lorenz distance',fontsize=11)

plt.subplot(2,3,5,xticks=[0.02,0.07,0.12],ylim=(0,10))
plt.plot(urate_sensitivity[:,0],urate_sensitivity[:,4],'-k',linewidth=1)
plt.plot(0.07,fit,'.k',markersize=8)
plt.yticks([1,3,5,7,9],[])
plt.text(0.03,8.5,'Unemployment',fontsize=11)
plt.text(0.055,7.25,r'rate $\mho$',fontsize=11)

'''
plt.subplot(2,3,6,xticks=[0.004,0.008,0.012],xlim=(0.003,0.0125),ylim=(0,10))
plt.plot(mortality_sensitivity[:,0],mortality_sensitivity[:,4],'-k',linewidth=1)
plt.plot(0.00625,fit,'.k',markersize=8)
plt.yticks([1,3,5,7,9],[])
plt.text(0.0037,0.5,r'Mortality rate $\mathsf{D}$',fontsize=11)
'''

plt.subplot(2,3,6,xlim=(0.0,0.04),ylim=(0,10))
plt.plot(g_sensitivity[:,0],g_sensitivity[:,4],'-k',linewidth=1)
plt.plot(0.01,fit,'.k',markersize=8)
plt.xticks([0.0,0.02,0.04],['0','0.02','0.04'])
plt.yticks([1,3,5,7,9],[])
plt.text(0.008,8.5,r'Aggregate',fontsize=11)
plt.text(0.005,7.25,r'growth rate $g$',fontsize=11)

plt.savefig(my_file_path  + '/Figures/FitSensitivity.pdf')
plt.show()
