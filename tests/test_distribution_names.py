"""
Tests for the lognormal classes' names: LogNormal and LogNormalMeanOne, and
their earlier names, Lognormal and MeanOneLogNormal, which still work, as the
classes under their new names, with a DeprecationWarning.
"""

import io
import pickle
import unittest
import warnings
from pathlib import Path

import numpy as np
import yaml

import HARK.distributions as distributions
import HARK.distributions.continuous as continuous
from HARK import parser
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.distributions import (
    LogNormal,
    LogNormalMeanOne,
    Lognormal,
    MeanOneLogNormal,
)

# The warning's message, for filters
RENAMED = r".* is now called "


def renamed_warnings(caught):
    """The renaming warnings among warnings caught by catch_warnings(record=True)."""
    return [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "is now called" in str(w.message)
    ]


# The default IndShockConsumerType's IncShkDstn[0], pickled on main at 4c285e86,
# before the rename, by pickle.dump(..., protocol=4) under Python 3.12, numpy
# 2.5.4, scipy 1.18.1 and xarray 2026.9.0. Its two continuous parents, kept in its
# record of the law's bounds, are stored as
# HARK.distributions.continuous.MeanOneLogNormal.
PICKLED_BEFORE_RENAME = (
    Path(__file__).parent / "data" / "income_distribution_before_rename.pkl"
)


class FixtureUnpickler(pickle.Unpickler):
    """Unpickles a fixture, skipping the test when the fixture names a path
    outside HARK that this environment lacks, such as an internal module of
    numpy, scipy or xarray that has since moved, and failing when the path is in
    HARK."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError) as err:
            if module.split(".")[0] == "HARK":
                raise
            raise unittest.SkipTest(
                f"the fixture names {module}.{name}, which this environment lacks"
            ) from err


class EarlierNamesTests(unittest.TestCase):
    def assert_warns_here(self, caught, old, new):
        # One warning, naming the new name, pointing at this test file
        found = renamed_warnings(caught)
        self.assertEqual(len(found), 1)
        message = str(found[0].message)
        self.assertIn(f"{old} is now called {new}", message)
        self.assertIn("not scheduled for removal", message)
        self.assertEqual(found[0].filename, __file__)

    def test_construction(self):
        for old, new, args in (
            (Lognormal, LogNormal, (0.1, 0.2)),
            (MeanOneLogNormal, LogNormalMeanOne, (0.2,)),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                dstn = old(*args, seed=7)
            self.assert_warns_here(caught, old.__name__, new.__name__)
            self.assertIs(type(dstn), new)
            np.testing.assert_array_equal(dstn.draw(20), new(*args, seed=7).draw(20))
        # with the keyword arguments of the alternative parametrization
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            dstn = Lognormal(mean=1.3, std=0.4)
        self.assertEqual(dstn.mu, LogNormal(mean=1.3, std=0.4).mu)

    def test_isinstance_and_issubclass(self):
        new_object, mean_one = LogNormal(0.1, 0.2), LogNormalMeanOne(0.2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertTrue(isinstance(new_object, Lognormal))
        self.assert_warns_here(caught, "Lognormal", "LogNormal")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertTrue(isinstance(mean_one, Lognormal))
            self.assertTrue(isinstance(mean_one, MeanOneLogNormal))
            self.assertFalse(isinstance(new_object, MeanOneLogNormal))
            self.assertTrue(issubclass(LogNormal, Lognormal))
            self.assertTrue(issubclass(LogNormalMeanOne, Lognormal))
            self.assertTrue(issubclass(LogNormalMeanOne, MeanOneLogNormal))
            self.assertFalse(issubclass(LogNormal, MeanOneLogNormal))
            old_object = Lognormal(0.1, 0.2)
        self.assertTrue(isinstance(old_object, LogNormal))
        # the old names are subclasses of the new ones, in either direction
        self.assertTrue(issubclass(Lognormal, LogNormal))
        self.assertTrue(issubclass(MeanOneLogNormal, LogNormalMeanOne))
        self.assertTrue(issubclass(MeanOneLogNormal, LogNormal))
        self.assertTrue(issubclass(Lognormal, continuous.ContinuousFrozenDistribution))

    def test_subclassing(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            class MyShock(MeanOneLogNormal):
                def __init__(self, sigma=1.0, seed=0):
                    super().__init__(sigma=sigma, seed=seed)
                    self.label = "mine"

        self.assert_warns_here(caught, "MeanOneLogNormal", "LogNormalMeanOne")
        self.assertIs(MyShock.__mro__[1], LogNormalMeanOne)
        self.assertIs(type(MyShock), type)
        shock = MyShock(0.3, seed=2)
        self.assertIs(type(shock), MyShock)
        self.assertEqual(shock.label, "mine")
        np.testing.assert_array_equal(
            shock.draw(10), LogNormalMeanOne(0.3, seed=2).draw(10)
        )

    def test_class_attributes_and_methods(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dstn = Lognormal.from_mean_std(1.0, 0.1, seed=3)
        self.assert_warns_here(caught, "Lognormal", "LogNormal")
        self.assertIs(type(dstn), LogNormal)
        np.testing.assert_array_equal(
            dstn.draw(10), LogNormal.from_mean_std(1.0, 0.1, seed=3).draw(10)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            method = Lognormal._approx_equiprobable
        self.assert_warns_here(caught, "Lognormal", "LogNormal")
        self.assertIs(method, LogNormal._approx_equiprobable)

    def test_setting_a_method_through_the_old_name(self):
        # Code that patches a method through the old name patches the class itself
        original = LogNormal._approx_equiprobable

        def patched(self, N, *args, **kwargs):
            return "patched"

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                Lognormal._approx_equiprobable = patched
            self.assert_warns_here(caught, "Lognormal", "LogNormal")
            self.assertIs(LogNormal._approx_equiprobable, patched)
            self.assertEqual(LogNormal(0.0, 0.1)._approx_equiprobable(3), "patched")
            self.assertEqual(LogNormalMeanOne(0.1)._approx_equiprobable(3), "patched")
        finally:
            LogNormal._approx_equiprobable = original
        self.assertNotIn("_approx_equiprobable", vars(Lognormal))

    def test_adding_and_deleting_an_attribute_through_the_old_name(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Lognormal.extra_attribute = 5
        self.assert_warns_here(caught, "Lognormal", "LogNormal")
        self.assertEqual(LogNormal.extra_attribute, 5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            del Lognormal.extra_attribute
        self.assert_warns_here(caught, "Lognormal", "LogNormal")
        self.assertFalse(hasattr(LogNormal, "extra_attribute"))

    def test_imports(self):
        self.assertIs(distributions.Lognormal, continuous.Lognormal)
        self.assertIs(distributions.MeanOneLogNormal, continuous.MeanOneLogNormal)
        for name in ("LogNormal", "LogNormalMeanOne", "Lognormal", "MeanOneLogNormal"):
            self.assertIn(name, distributions.__all__)

    def test_star_import(self):
        # A star import binds both names without warning; a later use of an old
        # name warns
        namespace = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec("from HARK.distributions import *", namespace)
        self.assertEqual(renamed_warnings(caught), [])
        self.assertIs(namespace["Lognormal"], Lognormal)
        self.assertIs(namespace["LogNormal"], LogNormal)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            namespace["MeanOneLogNormal"](0.1)
        self.assertEqual(len(renamed_warnings(caught)), 1)

    def test_pickling(self):
        # An object pickled now names only the new class; one pickled under the
        # old name loads as an object of the new class
        for new, old, args in (
            (LogNormal, "Lognormal", (0.1, 0.2)),
            (LogNormalMeanOne, "MeanOneLogNormal", (0.2,)),
        ):
            dstn = new(*args, seed=5)
            stored = pickle.dumps(dstn)
            self.assertIn(new.__name__.encode(), stored)
            self.assertNotIn(old.encode(), stored)
            # the same object as pickled when the class had its old name: the
            # names have the same length, so the stream is otherwise unchanged
            name = new.__name__.encode()
            stored_old = stored.replace(
                bytes([0x8C, len(name)]) + name,
                bytes([0x8C, len(old)]) + old.encode(),
            )
            self.assertNotEqual(stored_old, stored)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = pickle.loads(stored_old)
            self.assertEqual(renamed_warnings(caught), [])
            self.assertIs(type(loaded), new)
            np.testing.assert_array_equal(loaded.draw(10), new(*args, seed=5).draw(10))
        # the old name itself pickles as a reference to it
        self.assertIs(pickle.loads(pickle.dumps(Lognormal)), Lognormal)

    def test_unpickling_an_income_distribution_pickled_before_the_rename(self):
        stored = PICKLED_BEFORE_RENAME.read_bytes()
        self.assertIn(b"MeanOneLogNormal", stored)
        self.assertNotIn(b"LogNormalMeanOne", stored)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = FixtureUnpickler(io.BytesIO(stored)).load()
        self.assertEqual(renamed_warnings(caught), [])

        # Both continuous parents load as objects of the class under its new name
        fresh = IndShockConsumerType().IncShkDstn[0]
        self.assertIs(type(loaded), type(fresh))
        for get_parent in (
            lambda dstn: dstn.limit["dist"][0].limit["dist"],
            lambda dstn: dstn.limit["dist"][1].limit["dist"].limit["dist"],
        ):
            parent, fresh_parent = get_parent(loaded), get_parent(fresh)
            self.assertIs(type(parent), LogNormalMeanOne)
            self.assertIs(type(fresh_parent), LogNormalMeanOne)
            np.testing.assert_array_equal(parent.mu, fresh_parent.mu)
            np.testing.assert_array_equal(parent.sigma, fresh_parent.sigma)

        # and the distribution is the one a fresh build makes
        np.testing.assert_array_equal(loaded.pmv, fresh.pmv)
        np.testing.assert_array_equal(loaded.atoms, fresh.atoms)
        for bound in ("infimum", "supremum"):
            np.testing.assert_array_equal(loaded.limit[bound], fresh.limit[bound])
        np.testing.assert_array_equal(loaded.draw(10), fresh.draw(10))

    def test_once_per_call_site(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(3):
                Lognormal(0.1, 0.2)
            Lognormal(0.1, 0.2)
        self.assertEqual(len(renamed_warnings(caught)), 2)

    def test_new_names_never_warn(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message=RENAMED, category=DeprecationWarning
            )
            dstn = LogNormal(0.1, 0.2)
            LogNormalMeanOne(0.2).discretize(7)
            LogNormal.from_mean_std(1.0, 0.1)
            isinstance(dstn, LogNormal)
            issubclass(LogNormalMeanOne, LogNormal)
            LogNormal._approx_equiprobable

            class Mine(LogNormalMeanOne):
                pass

            pickle.loads(pickle.dumps(dstn))

    def test_yaml_tags(self):
        loader = parser.harklang_loader()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            new = yaml.load("a: !LogNormal {mean: 1.0, std: 0.2}", Loader=loader)
            mean_one = yaml.load("b: !LogNormalMeanOne {sigma: 0.2}", Loader=loader)
        self.assertEqual(
            [w for w in caught if issubclass(w.category, DeprecationWarning)], []
        )
        self.assertIs(new["a"][0], LogNormal)
        self.assertIs(mean_one["b"][0], LogNormalMeanOne)
        for tag, cls in (
            ("Lognormal", LogNormal),
            ("MeanOneLogNormal", LogNormalMeanOne),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                old = yaml.load(f"c: !{tag} {{sigma: 0.2}}", Loader=loader)
            found = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            self.assertEqual(len(found), 1)
            self.assertIn(f"!{tag} is now !{cls.__name__}", str(found[0].message))
            self.assertIs(old["c"][0], cls)
            self.assertEqual(old["c"][1], {"sigma": 0.2})


class NoEarlierNamesInsideHARKTests(unittest.TestCase):
    def test_solve_and_simulate_without_the_warning(self):
        # A full solve and simulation of the default model never uses an earlier
        # name: the warning would be an error here
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message=RENAMED, category=DeprecationWarning
            )
            agent = IndShockConsumerType(cycles=0)
            agent.solve()
            agent.T_sim = 20
            agent.AgentCount = 100
            agent.track_vars = ["aNrm", "cNrm", "pLvl"]
            agent.initialize_sim()
            agent.simulate()
        self.assertEqual(agent.history["cNrm"].shape, (20, 100))


if __name__ == "__main__":
    unittest.main()
