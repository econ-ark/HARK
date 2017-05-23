#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

/* Random number generator */
inline uint RNG(uint s) {
    uint seed = (s * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
    return (seed >> 0);
}



__kernel void solveConsIndShock(
    __global int *IntegerInputs
    ,__global int *TypeAddress
    ,__global int *Ttotal
    ,__global int *CoeffsAddress
    ,__global int *IncDstnAddress
    ,__global double *LivPrb
    ,__global double *IncomePrbs
    ,__global double *PermShks
    ,__global double *TranShks
    ,__global double *WorstIncPrb
    ,__global double *PermGroFac
    ,__global double *Rfree
    ,__global double *CRRA
    ,__global double *DiscFac
    ,__global double *BoroCnst
    ,__global double *aXtraGrid
    ,__global double *mGrid
    ,__global double *mLowerBound
    ,__global double *Coeffs0
    ,__global double *Coeffs1
    ,__global double *Coeffs2
    ,__global double *Coeffs3
    ,__global double *mTemp
    ,__global double *cTemp
    ,__global double *MPCtemp
    ,__global double *TestVar
) {

    /* Initialize this thread's id */
    int Gid = get_global_id(0);             /* global thread id */
    int Lid = get_local_id(0);              /* local thread id */
    int Nid = get_group_id(0);              /* workgroup number; each workgroup does one parameter set */
    if ((Nid >= IntegerInputs[1]) | (Lid >= IntegerInputs[7])) {
        return;
    }

    /* Unpack integer inputs */
    int aXtraCount = IntegerInputs[7];
    int GridSize = aXtraCount + 1;

    /* Get time-invariant data about this type */
    double R = Rfree[Nid];
    double rho = CRRA[Nid];
    double beta = DiscFac[Nid];
    double BoroCnstArt = BoroCnst[Nid];
    int TT = Ttotal[Nid];
    int LocA = TypeAddress[Nid];
    int temp = Nid*aXtraCount + Lid;
    double aNrmBase = aXtraGrid[temp];
    
    /* Initialize variables to be used during solution */
    int LocB;
    int LocC;
    int LocD;
    int IdxNext;
    int i;
    int IncomeDstnSize;
    double Gamma;
    double Dcancel;
    double wp; /* Weierstrauss p, worst income draw probability */
    double betaEff;
    double PatFac;
    double ExIncNext;
    double psi;
    double theta;
    double prob;
    double mNrm;
    double EndOfPrdvP;
    double EndOfPrdvPP;
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
    double aNrm;
    double cNrm;
    double MPC;
    double cNrmCons;
    double vP;
    double vPP;
    double dcda;
    double BoroCnstNat;
    double PermShkMin;
    double TranShkMin;
    double gap;
    double cBelow;
    double mBelow;
    double MPCbelow;
    double MPCabove;

    /* Trivially solve the terminal period t = TT-1 */
    double mBound = 0.0;
    double MPCmin = 1.0;
    double MPCmax = 1.0;
    double hNrm = 0.0;
    if (Lid == 0) {
        mLowerBound[LocA + TT-1] = 0.0;
    }
    int t = TT - 2;

    /* Loop over time, solving model by backward induction */
    while (t >= (0)) {
        /* Get buffer indices for this time period */
        LocB = LocA + t;
        LocC = CoeffsAddress[LocB] + Lid;
        LocD = IncDstnAddress[LocB];
        IncomeDstnSize = IncDstnAddress[LocB+1] - LocD;
        IdxNext = CoeffsAddress[LocB+1];

        /* Get data for this period */
        wp = WorstIncPrb[LocB];
        Gamma = PermGroFac[LocB+1];
        Dcancel = LivPrb[LocB+1];
        betaEff = beta*Dcancel;
        PatFac = powr(R*betaEff,1.0/rho)/R;
        MPCmin = 1.0/(1.0 + PatFac/MPCmin);
        MPCmax = 1.0/(1.0 + powr(wp,1.0/rho)*PatFac/MPCmax);
        PermShkMin = 1000000.0;
        TranShkMin = 1000000.0;
        ExIncNext = 0.0;

        /* Find the natural borrowing constraint */
        i = 0;
        while (i < IncomeDstnSize) {
            temp = LocD+i;
            prob = IncomePrbs[temp];
            psi = PermShks[temp];
            theta = TranShks[temp];
            ExIncNext = ExIncNext + psi*theta*prob;
            PermShkMin = (psi < PermShkMin) ? psi : PermShkMin;
            TranShkMin = (theta < TranShkMin) ? theta : TranShkMin;
            i += 1;
        }
        hNrm = Gamma/R*(ExIncNext + hNrm);
        BoroCnstNat = (mBound - TranShkMin)*Gamma*PermShkMin/R;
        
        /* Loop over future income shocks to calculate expected marginal value */
        aNrm = aNrmBase + BoroCnstNat;
        EndOfPrdvP = 0.0;
        EndOfPrdvPP = 0.0;
        i = 0;
        while (i < IncomeDstnSize) {
            /* Get market resources next period */
            temp = LocD+i;
            prob = IncomePrbs[temp];
            psi = PermShks[temp];
            theta = TranShks[temp];
            mNrm = R*aNrm/(Gamma*psi) + theta;

            /* Get consumption and the MPC next period */
            cNrm = mNrm; /* Default, which is kept if this is terminal period */
            MPC = 1.0;

            if (t < (TT-2)) {

            /* Find correct grid sector for this agent */
            Botj = 0;
            Topj = GridSize - 1;
            Botm = mGrid[IdxNext + Botj];
            Topm = mGrid[IdxNext + Topj];
            Diffj = Topj - Botj;
            Newj = Botj + Diffj/2;
            Newm = mGrid[IdxNext + Newj];
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
                    Newm = mGrid[IdxNext + Newj];
                }
                j = Botj;
            }
    
            /* Get the interpolation coefficients for this segment */
            temp = IdxNext + j;
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
            cNrmCons = mNrm - mBound;
            if (cNrmCons < cNrm) {
                cNrm = cNrmCons;
                MPC = 1.0;
            }
            } /* End of "if not terminal period" block */

            vP = powr(psi*cNrm,-rho);
            vPP = -rho*MPC*powr(psi*cNrm,-rho-1.0);
            EndOfPrdvP += prob*vP;
            EndOfPrdvPP += prob*vPP;
            if (i == 0) {
                TestVar[Gid] = (double)IdxNext;
            }
            i += 1;
        }

        /* Adjust end of period marginal value */
        EndOfPrdvP = betaEff*R*powr(Gamma,-rho)*EndOfPrdvP;
        EndOfPrdvPP = betaEff*R*R*powr(Gamma,-rho-1.0)*EndOfPrdvPP;

        /* Calculate consumption, the MPC, and the endogenous gridpoint */
        cNrm = powr(EndOfPrdvP,-1.0/rho);
        mNrm = aNrm + cNrm;
        dcda = EndOfPrdvPP/(-rho*powr(cNrm,-rho-1.0));
        MPC = dcda/(1.0 + dcda);

        /* Store results in shared memory for other threads to grab */
        mTemp[Gid] = mNrm;
        cTemp[Gid] = cNrm;
        MPCtemp[Gid] = MPC;

        mem_fence(CLK_GLOBAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE);

        /* Calculate coefficients for this segment */
        mBound = (BoroCnstArt > BoroCnstNat) ? BoroCnstArt : BoroCnstNat;
        mBelow = (Lid==0) ? BoroCnstNat : mTemp[Gid-1];
        cBelow = (Lid==0) ? 0.0 : cTemp[Gid-1];
        MPCbelow = (Lid==0) ? MPCmax : MPCtemp[Gid-1];
        Span = mNrm - mBelow;
        MPCbelow = MPCbelow*Span;
        MPCabove = MPC*Span;
        Coeffs0[LocC] = cBelow;
        Coeffs1[LocC] = MPCbelow;
        Coeffs2[LocC] = 3*(cNrm - cBelow) - 2*MPCbelow - MPCabove;
        Coeffs3[LocC] = MPCbelow + MPCabove + 2*(cBelow - cNrm);
        mGrid[LocC+1] = mNrm;
        if (Lid == 0) {
            mGrid[LocC] = BoroCnstNat;
        }

        /* Make the upper extrapolation coefficients */
        if (Lid == (aXtraCount-1)) {
            gap = MPCmin*mNrm + hNrm*MPCmin - cNrm;
            Coeffs0[LocC+1] = hNrm*MPCmin;
            Coeffs1[LocC+1] = MPCmin;
            Coeffs2[LocC+1] = gap;
            Coeffs3[LocC+1] = (MPCmin - MPC)/gap;
            mLowerBound[LocB] = mBound;
        }

        /* Advance time when all threads have completed this period */
        t += -1;
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE);

    } /* End of time loop */
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
