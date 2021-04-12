def add_Ex_values(self, stge_futr):
    self.crnt.PermShkValsNxtXref = PermShkValsNxtXref = stge_futr.IncShkDstnNxt.X[0]
    self.crnt.TranShkValsNxtXref = TranShkValsNxtXref = stge_futr.IncShkDstnNxt.X[1]
    self.crnt.ShkPrbsNext = ShkPrbsNext = self.crnt.IncShkPrbsNxt \
        = stge_futr.IncShkDstnNxt.pmf

    self.crnt.IncShkValsNxt = stge_futr.IncShkDstnNxt.X

    self.crnt.PermShkPrbsNxt = PermShkPrbsNxt = stge_futr.PermShkDstnNxt.pmf
    self.crnt.PermShkValsNxt = PermShkValsNxt = stge_futr.PermShkDstnNxt.X

    self.crnt.TranShkPrbsNxt = TranShkPrbsNxt = stge_futr.TranShkDstnNxt.pmf
    self.crnt.TranShkValsNxt = TranShkValsNxt = stge_futr.TranShkDstnNxt.X

    self.crnt.PermShkValsNxtMin = PermShkValsNxtMin = np.min(PermShkValsNxt)
    self.crnt.TranShkNxtMin = TranShkNxtMin = np.min(TranShkValsNxt)

    self.crnt.WorstIncPrbNxt = np.sum(
        ShkPrbsNext[
            (PermShkValsNxtXref * TranShkValsNxtXref)
            == (PermShkValsNxtMin * TranShkNxtMin)
        ]
    )

    # Precalc some useful expectations; search elsewhere for docs about them
    self.crnt.Inv_PermShkValsNxt = Inv_PermShkValsNxt = 1/PermShkValsNxt

    self.crnt.Ex_Inv_PermShk = \
        np.dot(Inv_PermShkValsNxt, PermShkPrbsNxt)

    self.crnt.Ex_uInv_PermShk = Ex_uInv_PermShk = \
        np.dot(PermShkValsNxt ** (1 - stge_futr.CRRA), PermShkPrbsNxt)

    self.crnt.uInv_Ex_uInv_PermShk = \
        Ex_uInv_PermShk ** (1 / (1 - stge_futr.CRRA))
