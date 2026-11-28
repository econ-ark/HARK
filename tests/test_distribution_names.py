"""
Tests for the lognormal classes' names: LogNormal, LogNormalMeanOne,
LogNormalTruncated and LogNormalTruncatedMeanOne, and their earlier names. Each
earlier name is the class under its new name, with a DeprecationWarning where it
is looked up in HARK.distributions or HARK.distributions.continuous.
"""

import abc
import functools
import inspect
import pickle
import unittest
import warnings

import numpy as np
import yaml

import HARK.distributions as distributions
import HARK.distributions.continuous as continuous
from HARK import parser
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.distributions import (
    LogNormal,
    LogNormalMeanOne,
    LogNormalTruncated,
    LogNormalTruncatedMeanOne,
)

# Each earlier name, the class it names now, and arguments to make one
EARLIER = (
    ("Lognormal", LogNormal, (0.1, 0.2)),
    ("MeanOneLogNormal", LogNormalMeanOne, (0.2,)),
    ("TruncatedLognormal", LogNormalTruncated, (0.1, 0.2)),
    ("TruncatedLogNormal", LogNormalTruncated, (0.1, 0.2)),
    ("MeanOneTruncatedLogNormal", LogNormalTruncatedMeanOne, (0.2,)),
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


def look_up(module, name):
    """The name looked up in module, and the renaming warnings that raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = getattr(module, name)
    return found, renamed_warnings(caught)


def as_pickled_under(stored, new, old, protocol):
    """
    The pickle stored, which names the class new, as it was pickled when the
    class was called old. Each old name has its new name's length, so only the
    name changes: in a GLOBAL opcode's text below protocol 4, in a
    SHORT_BINUNICODE string from it.
    """
    new, old = new.encode(), old.encode()
    assert len(new) == len(old)
    if protocol >= 4:
        return stored.replace(
            bytes([0x8C, len(new)]) + new, bytes([0x8C, len(old)]) + old
        )
    return stored.replace(b"\n" + new + b"\n", b"\n" + old + b"\n")


class EarlierNamesTests(unittest.TestCase):
    def test_each_earlier_name_is_the_class(self):
        for old, new, args in EARLIER:
            for module in (distributions, continuous):
                found, caught = look_up(module, old)
                self.assertIs(found, new)
                # one warning, naming the new name, at the line that looked it up
                self.assertEqual(len(caught), 1)
                message = str(caught[0].message)
                self.assertIn(f"{old} is now called {new.__name__}", message)
                self.assertIn("not scheduled for removal", message)
                self.assertEqual(caught[0].filename, __file__)
            # an object it makes is of that class
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                dstn = found(*args, seed=7)
                self.assertIs(type(dstn), found)
                self.assertTrue(isinstance(dstn, found))
            self.assertEqual(renamed_warnings(caught), [])
            np.testing.assert_array_equal(dstn.draw(20), new(*args, seed=7).draw(20))

    def test_an_import_warns_where_it_is_written_and_its_uses_do_not(self):
        # The import machinery looks the name up before the import statement
        # does; both lookups warn at this line, so the default filter shows one
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            from HARK.distributions import MeanOneLogNormal
        found = renamed_warnings(caught)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].filename, __file__)
        self.assertIs(MeanOneLogNormal, LogNormalMeanOne)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dstn = MeanOneLogNormal(0.2)
            self.assertTrue(isinstance(dstn, MeanOneLogNormal))
            self.assertTrue(issubclass(LogNormalMeanOne, MeanOneLogNormal))
            MeanOneLogNormal.from_mean_std
        self.assertEqual(renamed_warnings(caught), [])

    def test_the_class_itself(self):
        Lognormal, _ = look_up(distributions, "Lognormal")
        # its signature, and no alias among the new classes' subclasses
        self.assertEqual(inspect.signature(Lognormal), inspect.signature(LogNormal))
        earlier = {old for old, _, _ in EARLIER}
        for new in (LogNormal, LogNormalMeanOne, LogNormalTruncated):
            self.assertFalse({c.__name__ for c in new.__subclasses__()} & earlier)
        # a subclass that mixes in a base with another metaclass
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            class Mixed(Lognormal, abc.ABC):
                pass

            # dispatch on the class, and the class as a key
            @functools.singledispatch
            def kind(x):
                return "other"

            @kind.register
            def _(x: LogNormal):
                return "lognormal"

            dstn = Lognormal(0.1, 0.2)
            self.assertEqual(kind(dstn), "lognormal")
            self.assertEqual({LogNormal: "found"}[type(dstn)], "found")
        self.assertEqual(renamed_warnings(caught), [])
        self.assertIs(type(Mixed), abc.ABCMeta)
        self.assertIs(Mixed.__mro__[1], LogNormal)

    def test_star_import_binds_only_the_new_names(self):
        namespace = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec("from HARK.distributions import *", namespace)
        self.assertEqual(renamed_warnings(caught), [])
        for old, new, _ in EARLIER:
            self.assertNotIn(old, distributions.__all__)
            self.assertNotIn(old, namespace)
            self.assertIs(namespace[new.__name__], new)

    def test_other_names_are_still_missing(self):
        with self.assertRaises(AttributeError):
            distributions.NoSuchDistribution
        with self.assertRaises(AttributeError):
            continuous.NoSuchDistribution
        with self.assertRaises(ImportError):
            from HARK.distributions import NoSuchDistribution  # noqa: F401

    def test_pickled_under_an_earlier_name_at_every_protocol(self):
        for old, new, args in EARLIER:
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                stored = pickle.dumps(new(*args, seed=6), protocol=protocol)
                # an object pickled now names only the class's new name
                self.assertNotIn(old.encode(), stored)
                stored_old = as_pickled_under(stored, new.__name__, old, protocol)
                self.assertNotEqual(stored_old, stored)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    loaded = pickle.loads(stored_old)
                # loading looks the old name up, and so warns
                self.assertTrue(renamed_warnings(caught))
                self.assertIs(type(loaded), new)
                np.testing.assert_array_equal(
                    loaded.draw(10), new(*args, seed=6).draw(10)
                )

    def test_an_income_distribution_pickled_before_the_rename(self):
        # IndShockConsumerType's IncShkDstn[0] with unbounded lognormal shocks, the
        # default before the bounded laws, keeps its two continuous parents,
        # LogNormalMeanOne, in its record of the law's bounds; before the rename a
        # pickle stored them as MeanOneLogNormal. None now takes the default
        # bounds, so the build asks for no bounds with infinite ones
        unbounded = {
            "PermShk_inf": -np.inf,
            "PermShk_sup": np.inf,
            "TranShk_inf": -np.inf,
            "TranShk_sup": np.inf,
        }
        fresh = IndShockConsumerType(**unbounded).IncShkDstn[0]
        parents = (
            lambda dstn: dstn.limit["dist"][0].limit["dist"],
            lambda dstn: dstn.limit["dist"][1].limit["dist"].limit["dist"],
        )
        # (protocols 0 and 1 cannot pickle it: parts of it define __slots__)
        for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
            stored = pickle.dumps(fresh, protocol=protocol)
            stored_old = as_pickled_under(
                stored, "LogNormalMeanOne", "MeanOneLogNormal", protocol
            )
            self.assertIn(b"MeanOneLogNormal", stored_old)
            self.assertNotIn(b"LogNormalMeanOne", stored_old)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                loaded = pickle.loads(stored_old)
            self.assertIs(type(loaded), type(fresh))
            for parent in parents:
                self.assertIs(type(parent(loaded)), LogNormalMeanOne)
                np.testing.assert_array_equal(parent(loaded).mu, parent(fresh).mu)
                np.testing.assert_array_equal(parent(loaded).sigma, parent(fresh).sigma)
            np.testing.assert_array_equal(loaded.pmv, fresh.pmv)
            np.testing.assert_array_equal(loaded.atoms, fresh.atoms)
            for bound in ("infimum", "supremum"):
                np.testing.assert_array_equal(loaded.limit[bound], fresh.limit[bound])

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
            LogNormalTruncatedMeanOne(0.2).discretize(7)

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
            # at the line that loaded the YAML, not inside PyYAML
            self.assertEqual(found[0].filename, __file__)
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
