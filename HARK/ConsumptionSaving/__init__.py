__all__ = [
    "PerfForesightConsumerType",
    "IndShockConsumerType",
    "KinkedRconsumerType",
    "AggShockConsumerType",
    "AggShockMarkovConsumerType",
    "CobbDouglasEconomy",
    "SmallOpenEconomy",
    "CobbDouglasMarkovEconomy",
    "SmallOpenMarkovEconomy",
    "GenIncProcessConsumerType",
    "IndShockExplicitPermIncConsumerType",
    "PersistentShockConsumerType",
    "MarkovConsumerType",
    "MedExtMargConsumerType",
    "MedShockConsumerType",
    "PortfolioConsumerType",
    "PrefShockConsumerType",
    "KinkyPrefConsumerType",
    "RiskyAssetConsumerType",
    "RepAgentConsumerType",
    "RepAgentMarkovConsumerType",
    "TractableConsumerType",
    "BequestWarmGlowConsumerType",
    "BequestWarmGlowPortfolioType",
    "WealthUtilityConsumerType",
    "WealthPortfolioConsumerType",
    "LaborIntMargConsumerType",
    "BasicHealthConsumerType",
    "RiskyContribConsumerType",
    "IndShockConsumerTypeFast",
    "PerfForesightConsumerTypeFast",
    "HabitConsumerType",
    "HabitPortfolioConsumerType",
]

from HARK.ConsumptionSaving.ConsIndShockModel import (
    PerfForesightConsumerType,
    IndShockConsumerType,
    KinkedRconsumerType,
)
from HARK.ConsumptionSaving.ConsAggShockModel import (
    AggShockConsumerType,
    AggShockMarkovConsumerType,
    CobbDouglasEconomy,
    CobbDouglasMarkovEconomy,
    SmallOpenEconomy,
    SmallOpenMarkovEconomy,
)
from HARK.ConsumptionSaving.ConsGenIncProcessModel import (
    GenIncProcessConsumerType,
    IndShockExplicitPermIncConsumerType,
    PersistentShockConsumerType,
)
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.ConsMedModel import (
    MedExtMargConsumerType,
    MedShockConsumerType,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import PortfolioConsumerType
from HARK.ConsumptionSaving.ConsPrefShockModel import (
    PrefShockConsumerType,
    KinkyPrefConsumerType,
)
from HARK.ConsumptionSaving.ConsRepAgentModel import (
    RepAgentConsumerType,
    RepAgentMarkovConsumerType,
)
from HARK.ConsumptionSaving.TractableBufferStockModel import TractableConsumerType
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyAssetConsumerType
from HARK.ConsumptionSaving.ConsBequestModel import (
    BequestWarmGlowConsumerType,
    BequestWarmGlowPortfolioType,
)
from HARK.ConsumptionSaving.ConsWealthUtilityModel import WealthUtilityConsumerType
from HARK.ConsumptionSaving.ConsWealthPortfolioModel import WealthPortfolioConsumerType
from HARK.ConsumptionSaving.ConsLaborModel import LaborIntMargConsumerType
from HARK.ConsumptionSaving.ConsHealthModel import BasicHealthConsumerType
from HARK.ConsumptionSaving.ConsRiskyContribModel import RiskyContribConsumerType

try:
    from HARK.ConsumptionSaving.ConsIndShockModelFast import (
        IndShockConsumerTypeFast,
        PerfForesightConsumerTypeFast,
    )
except ImportError as _fast_exc:  # pragma: no cover - only where numba is absent
    # The Fast variants need third-party `interpolation` and `quantecon`, which
    # import numba themselves, so HARK._numba cannot cover them. Keep the names
    # bound so `import HARK.ConsumptionSaving` still works, and explain on use.
    def _fast_unavailable(*args, **kwargs):
        raise ImportError(
            "The Fast consumer types require the 'interpolation' and 'quantecon' "
            "packages, which import numba. numba is unavailable on this platform "
            "(for example stock Pyodide), so use IndShockConsumerType or "
            "PerfForesightConsumerType instead."
        ) from _fast_exc

    IndShockConsumerTypeFast = _fast_unavailable
    PerfForesightConsumerTypeFast = _fast_unavailable
from HARK.ConsumptionSaving.ConsHabitModel import (
    HabitConsumerType,
    HabitPortfolioConsumerType,
)
