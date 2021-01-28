# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:04:24 2021

@author: Mateo
"""

# Bring in modules we need
import unittest
import numpy as np
from HARK.Calibration.Income.SabelhausSongProfiles import sabelhaus_song_var_profile


class test_SabelhausSongProfiles(unittest.TestCase):
    def setUp(self):

        # Write results from Figure 6 in the original paper
        self.Fig6Coh1940Tran = np.array(
            [
                0.1288,
                0.1353,
                0.1386,
                0.1388,
                0.1392,
                0.1390,
                0.1365,
                0.1361,
                0.1346,
                0.1311,
                0.1287,
                0.1254,
                0.1228,
                0.1194,
                0.1175,
                0.1131,
                0.1096,
                0.1055,
                0.1024,
                0.0983,
                0.0949,
                0.0899,
                0.0838,
                0.0794,
                0.0763,
                0.0711,
                0.0642,
                0.0575,
            ]
        )

        self.Fig6Coh1940Perm = np.array(
            [
                0.0754,
                0.0623,
                0.0555,
                0.0520,
                0.0471,
                0.0449,
                0.0432,
                0.0403,
                0.0372,
                0.0373,
                0.0334,
                0.0337,
                0.0308,
                0.0312,
                0.0286,
                0.0304,
                0.0282,
                0.0283,
                0.0281,
                0.0275,
                0.0265,
                0.0279,
                0.0290,
                0.0289,
                0.0318,
                0.0332,
                0.0351,
                0.0383,
            ]
        )

        self.Fig6Coh1965Tran = np.array(
            [
                0.1066,
                0.1131,
                0.1164,
                0.1166,
                0.1170,
                0.1168,
                0.1143,
                0.1139,
                0.1124,
                0.1089,
                0.1065,
                0.1032,
                0.1006,
                0.0972,
                0.0953,
                0.0909,
                0.0874,
                0.0833,
                0.0802,
                0.0761,
                0.0727,
                0.0677,
                0.0616,
                0.0572,
                0.0541,
                0.0489,
                0.0420,
                0.0352,
            ]
        )
        self.Fig6Coh1965Perm = np.array(
            [
                0.0605,
                0.0474,
                0.0406,
                0.0371,
                0.0322,
                0.0300,
                0.0283,
                0.0253,
                0.0223,
                0.0224,
                0.0185,
                0.0187,
                0.0159,
                0.0163,
                0.0137,
                0.0154,
                0.0133,
                0.0134,
                0.0132,
                0.0126,
                0.0116,
                0.0130,
                0.0141,
                0.0140,
                0.0169,
                0.0183,
                0.0202,
                0.0234,
            ]
        )

        # Aggregate result from Sabelhaus' excel file
        self.AggTran = np.array(
            [
                0.1062,
                0.1136,
                0.1177,
                0.1188,
                0.1202,
                0.1208,
                0.1192,
                0.1197,
                0.1191,
                0.1164,
                0.1150,
                0.1126,
                0.1108,
                0.1083,
                0.1073,
                0.1038,
                0.1012,
                0.0979,
                0.0958,
                0.0925,
                0.0900,
                0.0859,
                0.0807,
                0.0772,
                0.0750,
                0.0706,
                0.0647,
                0.0588,
            ]
        )
        self.AggPerm = np.array(
            [
                0.0599,
                0.0474,
                0.0412,
                0.0383,
                0.0340,
                0.0324,
                0.0312,
                0.0289,
                0.0264,
                0.0272,
                0.0238,
                0.0247,
                0.0224,
                0.0235,
                0.0215,
                0.0238,
                0.0222,
                0.0230,
                0.0233,
                0.0233,
                0.0229,
                0.0250,
                0.0266,
                0.0271,
                0.0306,
                0.0326,
                0.0351,
                0.0389,
            ]
        )

    def test_paper_results(self):
        # Test own function against the profiles from Figure 6 in the paper

        # 1940 cohort
        stds1940 = sabelhaus_song_var_profile(
            age_min=27, age_max=54, cohort=1940, smooth=False
        )

        self.assertTrue(
            np.allclose(
                self.Fig6Coh1940Tran, np.array(stds1940["TranShkStd"]), atol=1e-03
            )
        )

        self.assertTrue(
            np.allclose(
                self.Fig6Coh1940Perm, np.array(stds1940["PermShkStd"]), atol=1e-03
            )
        )

        # 1965 cohort
        stds1965 = sabelhaus_song_var_profile(
            age_min=27, age_max=54, cohort=1965, smooth=False
        )

        self.assertTrue(
            np.allclose(
                self.Fig6Coh1965Tran, np.array(stds1965["TranShkStd"]), atol=1e-03
            )
        )

        self.assertTrue(
            np.allclose(
                self.Fig6Coh1965Perm, np.array(stds1965["PermShkStd"]), atol=1e-03
            )
        )

    def test_aggregate_results(self):
        # Tests own function against the aggregate profiles provided by Sabelhaus
        stds_agg = sabelhaus_song_var_profile(
            age_min=27, age_max=54, cohort=None, smooth=False
        )

        self.assertTrue(
            np.allclose(self.AggTran, np.array(stds_agg["TranShkStd"]), atol=1e-03)
        )

        self.assertTrue(
            np.allclose(self.AggPerm, np.array(stds_agg["PermShkStd"]), atol=1e-03)
        )

    def test_smoothing(self):
        # Tests the smoothing approximations that we make. The test is wether
        # the smoothed profiles are "close" to the exact estimates from Sabelhaus

        # Ensure smoothed versions are within +-10% of the exact estimates
        rtol = 1e-1

        smooth1940 = sabelhaus_song_var_profile(
            age_min=27, age_max=54, cohort=1940, smooth=True
        )
        smooth1965 = sabelhaus_song_var_profile(
            age_min=27, age_max=54, cohort=1965, smooth=True
        )
        smoothAgg = sabelhaus_song_var_profile(
            age_min=27, age_max=54, cohort=None, smooth=True
        )

        # 1940
        self.assertTrue(
            np.allclose(
                np.array(smooth1940["TranShkStd"]), self.Fig6Coh1940Tran, rtol= rtol
            )
        )

        self.assertTrue(
            np.allclose(
                np.array(smooth1940["PermShkStd"]), self.Fig6Coh1940Perm, atol= rtol
            )
        )
        
        # 1965
        self.assertTrue(
            np.allclose(
                np.array(smooth1965["TranShkStd"]), self.Fig6Coh1965Tran, rtol= rtol
            )
        )

        self.assertTrue(
            np.allclose(
                np.array(smooth1965["PermShkStd"]), self.Fig6Coh1965Perm, atol= rtol
            )
        )
        
        # Aggregate
        self.assertTrue(
            np.allclose(
                np.array(smoothAgg["TranShkStd"]), self.AggTran, rtol= rtol
            )
        )

        self.assertTrue(
            np.allclose(
                np.array(smoothAgg["PermShkStd"]), self.AggPerm, atol= rtol
            )
        )
