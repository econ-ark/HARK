#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

/* Random number generator */
inline uint RNG(uint s) {
    uint seed = (s * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
    return (seed >> 0);
}




/* Kernel for killing off agents and replacing them in ConsIndShockModel */
__kernel void getMortality(
     __global int *IntegerInputs
    ,__global int *TypeNow
    ,__global int *tCycleNow
    ,__global int *tAgeNow
    ,__global int *TypeAddress
    ,__global double *NormDraws
    ,__global double *UniDraws
    ,__global double *LivPrb
    ,__global double *aNrmInitMean
    ,__global double *aNrmInitStd
    ,__global double *pLvlInitMean
    ,__global double *pLvlInitStd
    ,__global double *aNrmNow
    ,__global double *pLvlNow
) {

    /* Initialize this thread's id */
    int Gid = get_global_id(0);             /* global thread id */
    if (Gid >= IntegerInputs[0]){
        return;
    }

    /* Unpack the integer inputs */
    int AgentCount = IntegerInputs[0];
    int tSim = IntegerInputs[4];
    int NormCount = IntegerInputs[5];
    int UniCount = IntegerInputs[6];
    double aNrm;
    double pLvl;

    /* Get basic information about this agent */
    int Type = TypeNow[Gid];
    int LocA = TypeAddress[Type];
    int temp = LocA + tCycleNow[Gid];
    
    /* Randomly draw whether this agent should be replaced */
    uint Seed = (uint)(tSim*AgentCount + Gid + 15);
    uint LivRand = RNG(Seed);
    LivRand = LivRand - UniCount*(LivRand/UniCount);
    double LivShk = UniDraws[LivRand];
    if (LivShk > LivPrb[temp]) {
	uint pRand = RNG(Seed+1);
	uint aRand = RNG(Seed+2);
        pRand = pRand - NormCount*(pRand/NormCount);
        aRand = aRand - NormCount*(aRand/NormCount);
        pLvl = exp(NormDraws[pRand]*pLvlInitStd[Type] + pLvlInitMean[Type]);
        aNrm = exp(NormDraws[aRand]*aNrmInitStd[Type] + aNrmInitMean[Type]);
        pLvlNow[Gid] = pLvl;
        aNrmNow[Gid] = aNrm;
        tCycleNow[Gid] = 0;
        tAgeNow[Gid] = 0;
    }
}




/* Kernel for obtaining shock variables in ConsIndShockModel */
__kernel void getShocks(
     __global int *IntegerInputs
    ,__global int *TypeNow
    ,__global int *tCycleNow
    ,__global int *TypeAddress
    ,__global double *NormDraws
    ,__global double *UniDraws
    ,__global double *PermStd
    ,__global double *TranStd
    ,__global double *UnempPrb
    ,__global double *IncUnemp
    ,__global double *PermShkNow
    ,__global double *TranShkNow
) {

    /* Initialize this thread's id */
    int Gid = get_global_id(0);             /* global thread id */
    if (Gid >= IntegerInputs[0]){
        return;
    }

    /* Unpack the integer inputs */
    int AgentCount = IntegerInputs[0];
    int tSim = IntegerInputs[4];
    int NormCount = IntegerInputs[5];
    int UniCount = IntegerInputs[6];

    /* Get basic information about this agent */
    int Type = TypeNow[Gid];
    int LocA = TypeAddress[Type];
    int temp = LocA + tCycleNow[Gid];
    
    /* Generate three random integers to be used */
    uint Seed = (uint)((tSim*AgentCount + Gid)*3);
    uint PermRand = RNG(Seed);
    uint TranRand = RNG(Seed+1);
    PermRand = PermRand - NormCount*(PermRand/NormCount);
    TranRand = TranRand - NormCount*(TranRand/NormCount);
    uint UnempRand = RNG(Seed+2);
    UnempRand = UnempRand - UniCount*(UnempRand/UniCount);

    /* Transform random integers into shocks for this agent */
    double psiStd = PermStd[temp];
    double thetaStd = TranStd[temp];
    double PermShk = exp(NormDraws[PermRand]*psiStd - 0.5*powr(psiStd,2.0));
    double TranShk = exp(NormDraws[TranRand]*thetaStd - 0.5*powr(thetaStd,2.0));
    double UnempShk = UniDraws[UnempRand];
    if (UnempShk < UnempPrb[temp]) {
        TranShk = IncUnemp[temp];
    }

    /* Store the shocks in global memory */
    PermShkNow[Gid] = PermShk;
    TranShkNow[Gid] = TranShk;
}





/* Kernel for calculating state variables at decision time in ConsIndShockModel */
__kernel void getStates(
     __global int *IntegerInputs
    ,__global int *TypeNow
    ,__global int *tCycleNow
    ,__global int *TypeAddress
    ,__global double *PermGroFac
    ,__global double *Rfree
    ,__global double *aNrmNow
    ,__global double *PermShkNow
    ,__global double *TranShkNow
    ,__global double *mNrmNow
    ,__global double *pLvlNow
) {

    /* Initialize this thread's id */
    int Gid = get_global_id(0);             /* global thread id */
    if (Gid >= IntegerInputs[0]){
        return;
    }

    /* Get consumer's type, permanent income growth factor, interest factor, and post-state */
    int Type = TypeNow[Gid];
    int Loc = TypeAddress[Type] + tCycleNow[Gid];
    double R = Rfree[Type];
    double Gamma = PermGroFac[Loc];
    double pLvl = pLvlNow[Gid];
    double aNrm = aNrmNow[Gid];

    /* Calculate consumer's market resources and new permanent income level */
    double psi = PermShkNow[Gid];
    pLvlNow[Gid] = pLvl*psi*Gamma;
    mNrmNow[Gid] = aNrm*R/(psi*Gamma) + TranShkNow[Gid];
}





/* Kernel for calculating the control variable in ConsIndShockModel */
__kernel void getControls(
     __global int *IntegerInputs
    ,__global int *TypeNow
    ,__global int *tCycleNow
    ,__global int *Ttotal
    ,__global int *TypeAddress
    ,__global int *CoeffsAddress
    ,__global double *mGrid
    ,__global double *mLowerBound
    ,__global double *Coeffs0
    ,__global double *Coeffs1
    ,__global double *Coeffs2
    ,__global double *Coeffs3
    ,__global double *mNrmNow
    ,__global double *cNrmNow
    ,__global double *MPCnow
) {

    /* Initialize this thread's id */
    int Gid = get_global_id(0);             /* global thread id */
    if (Gid >= IntegerInputs[0]){
        return;
    }

    /* Initialize some variables to be used shortly */
    int j;
    int Botj;
    int Topj;
    int Diffj;
    int Newj;
    double Botm;
    double Topm;
    double Newm;
    double b0;
    double b1;
    double b2;
    double b3;
    double Span;
    double mX;
    double cNrm;
    double MPC;

    /* Unpack the integer inputs */
    int TypeAgeSize = IntegerInputs[2];
    int CoeffsSize = IntegerInputs[3];

    /* Get basic information about this agent */
    int Type = TypeNow[Gid];
    int LocA = TypeAddress[Type];
    double mNrm = mNrmNow[Gid];
    int temp = LocA + tCycleNow[Gid];
    int LocB = CoeffsAddress[temp];
    int GridSize;
    if ((temp+1) == TypeAgeSize) {
        GridSize = CoeffsSize - LocB;
    }
    else {
        GridSize = CoeffsAddress[temp+1] - LocB;
    }
    double mBound = mLowerBound[LocB];

    cNrm = mNrm; /* Default, which is kept if this is terminal period */
    MPC = 1.0;

    if (tCycleNow[Gid] < (Ttotal[Type]-1)) {
    /* Find correct grid sector for this agent */
    Botj = 0;
    Topj = GridSize - 1;
    Botm = mGrid[LocB + Botj];
    Topm = mGrid[LocB + Topj];
    Diffj = Topj - Botj;
    Newj = Botj + Diffj/2;
    Newm = mGrid[LocB + Newj];
    if (mNrm < Botm) { /* If m is outside the grid bounds, this is easy (shouldn't happen) */
        j = 0;
        Topm = Botm;
        Botm = Topm - 1.0;
    }
    else if (mNrm > Topm) {
        j = GridSize-1;
        Botm = Topm;
    }
    else { /* Otherwise, perform a binary/golden search for the right segment */
        while (Diffj > 1) {
            if (mNrm < Newm) {
                Topj = Newj;
                Topm = Newm;
            }
            else {
                Botj = Newj;
                Botm = Newm;
            }
            Diffj = Topj - Botj;
            Newj = Botj + Diffj/2;
            Newm = mGrid[LocB + Newj];
        }
        j = Botj;
    }
    
    /* Get the interpolation coefficients for this segment */
    temp = LocB + j;
    b0 = Coeffs0[temp];
    b1 = Coeffs1[temp];
    b2 = Coeffs2[temp];
    b3 = Coeffs3[temp];
    if (Topm > Botm) {
        Span = (Topm - Botm);
    } else {
        Span = 1.0;
    }
    mX = (mNrm - Botm)/Span;

    /* Evaluate consumption on main portion of cFunc */
    if (j < (GridSize-1)) {
        cNrm = b0 + mX*(b1 + mX*(b2 + mX*(b3)));
        MPC = (b1 + mX*(2*b2 + mX*(3*b3)))/Span;
    }
    else { /* Evaluate consumption on extrapolated cFunc */
        cNrm = b0 + mNrm*b1 - b2*exp(mX*b3);
        MPC = b1 - b3*b2*exp(mX*b3);
    }

    /* Make sure consumption does not violate the borrowing constraint */
    double cNrmCons = mNrm - mBound;
    if (cNrmCons < cNrm) {
        cNrm = cNrmCons;
        MPC = 1.0;
    }
    } /* End of "if not terminal period" block */

    /* Store this agent's consumption and MPC in global buffers */
    cNrmNow[Gid] = cNrm;
    MPCnow[Gid] = MPC;
}





/* Kernel for calculating the post-decision state variables in ConsIndShockModel */
__kernel void getPostStates(
     __global int *IntegerInputs
    ,__global int *TypeNow
    ,__global int *tCycleNow
    ,__global int *tAgeNow
    ,__global int *Ttotal
    ,__global double *mNrmNow
    ,__global double *cNrmNow
    ,__global double *pLvlNow
    ,__global double *aNrmNow
    ,__global double *aLvlNow
) {

    /* Initialize this thread's id */
    int Gid = get_global_id(0);             /* global thread id */
    if (Gid >= IntegerInputs[0]){
        return;
    }

    /* Calculate end of period assets, normalized and in level */
    double aNrm = mNrmNow[Gid] - cNrmNow[Gid];
    aNrmNow[Gid] = aNrm;
    aLvlNow[Gid] = aNrm*pLvlNow[Gid];

    /* Advance time for this agent */
    int Type = TypeNow[Gid];
    tAgeNow[Gid] = tAgeNow[Gid] + 1;
    int temp = tCycleNow[Gid] + 1;
    if (temp == (Ttotal[Type])) {
        temp = 0;
    }
    tCycleNow[Gid] = temp;
}





/* All-in-one kernel for simulating one period of the consumption-saving model */
__kernel void simOnePeriod(
    __global int *IntegerInputs
    ,__global int *TypeNow
    ,__global int *tCycleNow
    ,__global int *tAgeNow
    ,__global int *Ttotal
    ,__global int *TypeAddress
    ,__global int *CoeffsAddress
    ,__global double *NormDraws
    ,__global double *UniDraws
    ,__global double *LivPrb
    ,__global double *aNrmInitMean
    ,__global double *aNrmInitStd
    ,__global double *pLvlInitMean
    ,__global double *pLvlInitStd
    ,__global double *PermStd
    ,__global double *TranStd
    ,__global double *UnempPrb
    ,__global double *IncUnemp
    ,__global double *PermGroFac
    ,__global double *Rfree
    ,__global double *mGrid
    ,__global double *mLowerBound
    ,__global double *Coeffs0
    ,__global double *Coeffs1
    ,__global double *Coeffs2
    ,__global double *Coeffs3
    ,__global double *PermShkNow
    ,__global double *TranShkNow
    ,__global double *aNrmNow
    ,__global double *pLvlNow
    ,__global double *mNrmNow
    ,__global double *cNrmNow
    ,__global double *MPCnow
    ,__global double *aLvlNow
    ,__global double *TestVar
) {

    /* Initialize this thread's id */
    int Gid = get_global_id(0);             /* global thread id */
    if (Gid >= IntegerInputs[0]){
        return;
    }

    /* Unpack the integer inputs */
    int AgentCount = IntegerInputs[0];
    int TypeCount = IntegerInputs[1]; /* unused */
    int TypeAgeSize = IntegerInputs[2];
    int CoeffsSize = IntegerInputs[3];
    int tSim = IntegerInputs[4];
    int NormCount = IntegerInputs[5];
    int UniCount = IntegerInputs[6];

    /* Get basic information about this agent */
    int Type = TypeNow[Gid];
    int LocA = TypeAddress[Type];
    int temp = LocA + tCycleNow[Gid];
    double R = Rfree[Type];

    double pLvl = pLvlNow[Gid];
    double aNrm = aNrmNow[Gid];
    

    /* Randomly draw whether this agent should be replaced */
    uint Seed = (uint)(tSim*AgentCount + Gid + 15);
    uint LivRand = RNG(Seed);
    LivRand = LivRand - UniCount*(LivRand/UniCount);
    double LivShk = UniDraws[LivRand];
    if (LivShk > LivPrb[temp]) {
	uint pRand = RNG(Seed+1);
	uint aRand = RNG(Seed+2);
        pRand = pRand - NormCount*(pRand/NormCount);
        aRand = aRand - NormCount*(aRand/NormCount);
        pLvl = exp(NormDraws[pRand]*pLvlInitStd[Type] + pLvlInitMean[Type]);
        aNrm = exp(NormDraws[aRand]*aNrmInitStd[Type] + aNrmInitMean[Type]);
        pLvlNow[Gid] = pLvl;
        aNrmNow[Gid] = aNrm;
        tCycleNow[Gid] = 0;
        tAgeNow[Gid] = 0;
        temp = LocA + 0;
    }

    /* Generate three random integers to be used */
    Seed = (uint)((tSim*AgentCount + Gid)*3);
    uint PermRand = RNG(Seed);
    uint TranRand = RNG(Seed+1);
    PermRand = PermRand - NormCount*(PermRand/NormCount);
    TranRand = TranRand - NormCount*(TranRand/NormCount);
    uint UnempRand = RNG(Seed+2);
    UnempRand = UnempRand - UniCount*(UnempRand/UniCount);

    /* Transform random integers into shocks for this agent */
    double psiStd = PermStd[temp];
    double thetaStd = TranStd[temp];
    double PermShk = exp(NormDraws[PermRand]*psiStd - 0.5*powr(psiStd,2.0));
    double TranShk = exp(NormDraws[TranRand]*thetaStd - 0.5*powr(thetaStd,2.0));
    double UnempShk = UniDraws[UnempRand];
    if (UnempShk < UnempPrb[temp]) {
        TranShk = IncUnemp[temp];
    }

    /* Store the shocks in global memory */
    PermShkNow[Gid] = PermShk;
    TranShkNow[Gid] = TranShk;

    /* Calculate consumer's market resources and new permanent income level */
    double Gamma = PermGroFac[temp];
    double psi = PermShk;
    pLvl = pLvl*psi*Gamma;
    double mNrm = aNrm*R/(psi*Gamma) + TranShk;
    pLvlNow[Gid] = pLvl;
    mNrmNow[Gid] = mNrm;

    /* Initialize some variables to be used shortly */
    int j;
    int Botj;
    int Topj;
    int Diffj;
    int Newj;
    double Botm;
    double Topm;
    double Newm;
    double b0;
    double b1;
    double b2;
    double b3;
    double Span;
    double mX;
    double cNrm;
    double MPC;

    /* Get some indices and bounds for this consumer's type-age */
    int LocB = CoeffsAddress[temp];
    int GridSize;
    if ((temp+1) == TypeAgeSize) {
        GridSize = CoeffsSize - LocB;
    }
    else {
        GridSize = CoeffsAddress[temp+1] - LocB;
    }
    double mBound = mLowerBound[LocB];

    cNrm = mNrm; /* Default, which is kept if this is terminal period */
    MPC = 1.0;

    if (tCycleNow[Gid] < (Ttotal[Type]-1)) {
    /* Find correct grid sector for this agent */
    Botj = 0;
    Topj = GridSize - 1;
    Botm = mGrid[LocB + Botj];
    Topm = mGrid[LocB + Topj];
    Diffj = Topj - Botj;
    Newj = Botj + Diffj/2;
    Newm = mGrid[LocB + Newj];
    if (mNrm < Botm) { /* If m is outside the grid bounds, this is easy (shouldn't happen) */
        j = 0;
        Topm = Botm;
        Botm = Topm - 1.0;
    }
    else if (mNrm > Topm) {
        j = GridSize-1;
        Botm = Topm;
    }
    else { /* Otherwise, perform a binary/golden search for the right segment */
        while (Diffj > 1) {
            if (mNrm < Newm) {
                Topj = Newj;
                Topm = Newm;
            }
            else {
                Botj = Newj;
                Botm = Newm;
            }
            Diffj = Topj - Botj;
            Newj = Botj + Diffj/2;
            Newm = mGrid[LocB + Newj];
        }
        j = Botj;
    }
    
    /* Get the interpolation coefficients for this segment */
    temp = LocB + j;
    b0 = Coeffs0[temp];
    b1 = Coeffs1[temp];
    b2 = Coeffs2[temp];
    b3 = Coeffs3[temp];
    if (Topm > Botm) {
        Span = (Topm - Botm);
    } else {
        Span = 1.0;
    }
    mX = (mNrm - Botm)/Span;

    /* Evaluate consumption on main portion of cFunc */
    if (j < (GridSize-1)) {
        cNrm = b0 + mX*(b1 + mX*(b2 + mX*(b3)));
        MPC = (b1 + mX*(2*b2 + mX*(3*b3)))/Span;
    }
    else { /* Evaluate consumption on extrapolated cFunc */
        cNrm = b0 + mNrm*b1 - b2*exp(mX*b3);
        MPC = b1 - b3*b2*exp(mX*b3);
    }

    /* Make sure consumption does not violate the borrowing constraint */
    double cNrmCons = mNrm - mBound;
    if (cNrmCons < cNrm) {
        cNrm = cNrmCons;
        MPC = 1.0;
    }
    } /* End of "if not terminal period" block */

    /* Store this agent's consumption and MPC in global buffers */
    cNrmNow[Gid] = cNrm;
    MPCnow[Gid] = MPC;

    /* Calculate end of period assets, normalized and in level */
    aNrm = mNrm - cNrm;
    aNrmNow[Gid] = aNrm;
    aLvlNow[Gid] = aNrm*pLvlNow[Gid];

    /* Advance time for this agent */
    TestVar[Gid] = tCycleNow[Gid];
    tAgeNow[Gid] = tAgeNow[Gid] + 1;
    temp = tCycleNow[Gid] + 1;
    if (temp == (Ttotal[Type])) {
        temp = 0;
    }
    tCycleNow[Gid] = temp;
}
