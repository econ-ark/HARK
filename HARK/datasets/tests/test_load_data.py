import unittest
from HARK.datasets import load_SCF_wealth_weights
from HARK.datasets.cpi.us.CPITools import cpi_deflator

class test_load_SCF_wealth_weights(unittest.TestCase):
    def setUp(self):
        self.SCF_wealth, self.SCF_weights = load_SCF_wealth_weights()

    def test_shape(self):
        self.assertEqual(self.SCF_wealth.shape, (3553,))
        self.assertEqual(self.SCF_weights.shape, (3553,))

# %% US CPI tests
class test_cpi_deflators(unittest.TestCase):
    
    def test_month_deflators(self):
        
        # Same year test
        defl_same_year = cpi_deflator(2000, 2000, 'SEP')
        self.assertEqual(defl_same_year[0], 1.0)
        
        # Different year test
        defl_diff_year = cpi_deflator(1998, 2019, 'SEP')
        self.assertAlmostEqual(defl_diff_year[0], 1.57279534)
        
    def test_avg_deflators(self):
        
        # Same year test
        defl_same_year = cpi_deflator(2000, 2000)
        self.assertEqual(defl_same_year[0], 1.0)
        
        # Different year test
        defl_diff_year = cpi_deflator(1998, 2019)
        self.assertAlmostEqual(defl_diff_year[0], 1.57202505)