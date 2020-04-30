from HARK.ConsumptionSaving.ConsIndShockModel import \
    IndShockConsumerType, ConsIndShockSolverBasic
import HARK.ConsumptionSaving.ConsumerParameters as Params

LifecycleExample = IndShockConsumerType(
    **Params.init_lifecycle)
LifecycleExample.cycles = 1
LifecycleExample.solve()

# test the solution_terminal
assert(
    LifecycleExample.solution[10].cFunc(2).tolist() == 2)

print(LifecycleExample.solution[9].cFunc(1))
assert(LifecycleExample.solution[9].cFunc(1) == 0.97769632)
self.assertAlmostEqual(LifecycleExample.solution[8].cFunc(1),
                       0.96624445)
self.assertAlmostEqual(LifecycleExample.solution[7].cFunc(1),
                       0.95691449)

self.assertAlmostEqual(
    LifecycleExample.solution[0].cFunc(1).tolist(),
    0.87362789)
self.assertAlmostEqual(
    LifecycleExample.solution[1].cFunc(1).tolist(),
    0.9081621)
self.assertAlmostEqual(
    LifecycleExample.solution[2].cFunc(1).tolist(),
    0.9563899)
