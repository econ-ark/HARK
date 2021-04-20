
class DolObj(MetricObject):
    def __init__(
            self,
            dsymbls=None,
            dsttes=None,
            dcntrols=None,
            dexpects=None,
            dvlues=None,
            dparms=None,
            drwards=None,
            ddefns=None,
            deqns={'darbitrge': dict(), 'dtrnstn': dict(), 'dvlues': dict(), 'dfelicty': dict(), 'ddirct_rspnse': dict()
                   },
            dcalibrtn=dict({'dparms': dict(), 'dendog': dict()}),
            dexog=dict(),
            ddmain=dict(),
            doptns=dict()
    ):
        self.about = {'DolObj': None}
