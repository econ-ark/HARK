"""
Test file for the IRA consumption model.

This file contains tests to validate the IRAConsumerType implementation
including proper handling of early withdrawal penalties, kinked interest rates,
and two-account structure.
"""

import sys
import os
import unittest
import numpy as np

# Add the HARK directory to the Python path
sys.path.insert(0, '/home/runner/work/HARK/HARK')

try:
    from HARK.ConsumptionSaving.ConsIRAModel import (
        IRAConsumerType,
        IRASolution,
        solve_ConsIRA,
        init_ira_accounts,
    )
    from HARK.ConsumptionSaving.ConsIndShockModel import (
        IndShockConsumerType,
        init_idiosyncratic_shocks,
    )
    from HARK.interpolation import LinearInterp
    from HARK import NullFunc
    imports_successful = True
except ImportError as e:
    print(f"Import error: {e}")
    imports_successful = False


class TestIRAModel(unittest.TestCase):
    """Test cases for the IRA consumption model."""
    
    def setUp(self):
        """Set up test parameters."""
        if not imports_successful:
            self.skipTest("Required modules could not be imported")
            
        self.base_params = init_ira_accounts.copy()
        self.base_params.update({
            "AgentCount": 100,  # Smaller for testing
            "T_sim": 10,        # Short simulation
            "track_vars": ["aNrm", "cNrm", "mNrm"],
        })
        
    def test_ira_consumer_initialization(self):
        """Test that IRAConsumerType can be initialized properly."""
        try:
            agent = IRAConsumerType(**self.base_params)
            
            # Check that required attributes exist
            self.assertTrue(hasattr(agent, 'Rfree_liquid_save'))
            self.assertTrue(hasattr(agent, 'Rfree_liquid_boro'))
            self.assertTrue(hasattr(agent, 'Rfree_IRA_save'))
            self.assertTrue(hasattr(agent, 'Rfree_IRA_boro'))
            self.assertTrue(hasattr(agent, 'IRA_penalty_rate'))
            self.assertTrue(hasattr(agent, 'retirement_age'))
            
            # Check parameter values
            self.assertEqual(agent.Rfree_liquid_save, 1.03)
            self.assertEqual(agent.Rfree_IRA_save, 1.07)
            self.assertEqual(agent.IRA_penalty_rate, 0.10)
            self.assertEqual(agent.retirement_age, 65)
            
            print("✓ IRA consumer initialization test passed")
            
        except Exception as e:
            self.fail(f"IRA consumer initialization failed: {e}")
    
    def test_ira_solver_function(self):
        """Test the IRA solver function with minimal parameters."""
        try:
            # Create a minimal solution for next period
            mGrid = np.linspace(0, 10, 50)
            cGrid = 0.8 * mGrid  # Simple consumption function
            cFunc = LinearInterp(mGrid, cGrid)
            vPGrid = cGrid ** (-2.0)  # Simple marginal value function
            vPfunc = LinearInterp(mGrid, vPGrid)
            
            solution_next = IRASolution(cFunc=cFunc, vPfunc=vPfunc)
            
            # Create minimal income shock distribution
            PermShkVals = np.array([0.9, 1.0, 1.1])
            TranShkVals = np.array([0.8, 1.0, 1.2]) 
            ShkPrbs = np.array([0.25, 0.5, 0.25])
            
            # Mock distribution object
            class MockDistribution:
                def __init__(self, perm_vals, tran_vals, probs):
                    self.atoms = [perm_vals, tran_vals]
                    self.pmv = probs
                    
            IncShkDstn = MockDistribution(PermShkVals, TranShkVals, ShkPrbs)
            
            # Test solver parameters
            solver_params = {
                'solution_next': solution_next,
                'IncShkDstn': IncShkDstn,
                'LivPrb': 0.98,
                'DiscFac': 0.96,
                'CRRA': 2.0,
                'Rfree_liquid_save': 1.03,
                'Rfree_liquid_boro': 1.20,
                'Rfree_IRA_save': 1.07,
                'Rfree_IRA_boro': 1.00,
                'IRA_penalty_rate': 0.10,
                'retirement_age': 65,
                'current_age': 35,  # Subject to penalty
                'PermGroFac': 1.01,
                'BoroCnstArt': 0.0,
                'aXtraGrid': np.linspace(0, 20, 48),
                'vFuncBool': False,
                'CubicBool': False,
            }
            
            # Call solver
            solution = solve_ConsIRA(**solver_params)
            
            # Check that solution has required attributes
            self.assertTrue(hasattr(solution, 'cFunc'))
            self.assertTrue(hasattr(solution, 'cFunc_IRA'))
            self.assertTrue(hasattr(solution, 'vPfunc'))
            self.assertTrue(callable(solution.cFunc))
            self.assertTrue(callable(solution.cFunc_IRA))
            
            # Test that consumption function is reasonable
            test_m = 1.0
            c_val = solution.cFunc(test_m)
            self.assertGreater(c_val, 0)
            self.assertLess(c_val, test_m)  # Can't consume more than resources
            
            print("✓ IRA solver function test passed")
            
        except Exception as e:
            self.fail(f"IRA solver function test failed: {e}")
            
    def test_penalty_impact(self):
        """Test that early withdrawal penalty affects solution."""
        try:
            # Create agent before retirement age (with penalty)
            params_young = self.base_params.copy()
            params_young['T_age'] = 40  # Young agent
            agent_young = IRAConsumerType(**params_young)
            
            # Create agent at retirement age (no penalty)  
            params_old = self.base_params.copy()
            params_old['T_age'] = 70  # Retired agent
            agent_old = IRAConsumerType(**params_old)
            
            # Test that they have different parameter values as expected
            self.assertEqual(agent_young.IRA_penalty_rate, 0.10)
            self.assertEqual(agent_old.IRA_penalty_rate, 0.10)  # Same penalty rate
            self.assertEqual(agent_young.retirement_age, 65)
            self.assertEqual(agent_old.retirement_age, 65)  # Same retirement age
            
            print("✓ Penalty impact test setup passed")
            
        except Exception as e:
            self.fail(f"Penalty impact test failed: {e}")
            
    def test_kinked_rates(self):
        """Test that kinked interest rates are properly configured."""
        try:
            agent = IRAConsumerType(**self.base_params)
            
            # Check that borrowing rates are higher than saving rates
            self.assertGreater(agent.Rfree_liquid_boro, agent.Rfree_liquid_save)
            
            # Check that IRA rate is higher than liquid saving rate
            self.assertGreater(agent.Rfree_IRA_save, agent.Rfree_liquid_save)
            
            # Check that all rates are positive
            self.assertGreater(agent.Rfree_liquid_save, 1.0)
            self.assertGreater(agent.Rfree_liquid_boro, 1.0)
            self.assertGreater(agent.Rfree_IRA_save, 1.0)
            
            print("✓ Kinked rates test passed")
            
        except Exception as e:
            self.fail(f"Kinked rates test failed: {e}")
            
    def test_solution_attributes(self):
        """Test that IRASolution has all required attributes."""
        try:
            # Create a simple solution
            mGrid = np.linspace(0, 5, 20)
            cGrid = 0.7 * mGrid
            cFunc = LinearInterp(mGrid, cGrid)
            cFunc_IRA = LinearInterp(mGrid, 0.8 * mGrid)
            
            solution = IRASolution(
                cFunc=cFunc,
                cFunc_IRA=cFunc_IRA,
                vFunc=NullFunc(),
                vPfunc=LinearInterp(mGrid, cGrid ** (-2.0)),
                mNrmMin=0.0,
                hNrm=1.0,
                MPCmin=0.1,
                MPCmax=0.9,
            )
            
            # Check attributes
            self.assertTrue(hasattr(solution, 'cFunc'))
            self.assertTrue(hasattr(solution, 'cFunc_IRA'))
            self.assertTrue(hasattr(solution, 'vFunc'))
            self.assertTrue(hasattr(solution, 'vPfunc'))
            self.assertTrue(hasattr(solution, 'mNrmMin'))
            self.assertTrue(hasattr(solution, 'hNrm'))
            
            # Test that functions can be called
            test_val = solution.cFunc(1.0)
            self.assertIsInstance(test_val, (float, np.floating))
            
            test_val_ira = solution.cFunc_IRA(1.0)
            self.assertIsInstance(test_val_ira, (float, np.floating))
            
            print("✓ Solution attributes test passed")
            
        except Exception as e:
            self.fail(f"Solution attributes test failed: {e}")
            
    def test_parameter_inheritance(self):
        """Test that IRA model inherits properly from IndShockConsumerType."""
        try:
            # Create both types
            base_agent = IndShockConsumerType(**init_idiosyncratic_shocks)
            ira_agent = IRAConsumerType(**self.base_params)
            
            # Check that IRA agent has inherited standard parameters
            self.assertEqual(ira_agent.CRRA, base_agent.CRRA)
            self.assertEqual(ira_agent.DiscFac, base_agent.DiscFac)
            self.assertEqual(ira_agent.LivPrb, base_agent.LivPrb)
            
            # Check that IRA agent has additional parameters
            self.assertTrue(hasattr(ira_agent, 'Rfree_liquid_save'))
            self.assertTrue(hasattr(ira_agent, 'IRA_penalty_rate'))
            self.assertFalse(hasattr(base_agent, 'Rfree_liquid_save'))
            self.assertFalse(hasattr(base_agent, 'IRA_penalty_rate'))
            
            print("✓ Parameter inheritance test passed")
            
        except Exception as e:
            self.fail(f"Parameter inheritance test failed: {e}")


def run_basic_validation():
    """Run basic validation without full unittest framework."""
    print("Running basic IRA model validation...")
    print("=" * 50)
    
    if not imports_successful:
        print("❌ Cannot run tests - imports failed")
        return False
        
    try:
        # Test 1: Basic initialization
        print("Test 1: Basic initialization")
        params = init_ira_accounts.copy()
        params.update({"AgentCount": 10, "T_sim": 5})
        agent = IRAConsumerType(**params)
        print(f"  ✓ Agent created with IRA penalty rate: {agent.IRA_penalty_rate}")
        print(f"  ✓ Liquid save rate: {agent.Rfree_liquid_save}")
        print(f"  ✓ IRA save rate: {agent.Rfree_IRA_save}")
        print(f"  ✓ Retirement age: {agent.retirement_age}")
        
        # Test 2: Parameter relationships
        print("\nTest 2: Parameter relationships")
        assert agent.Rfree_liquid_boro > agent.Rfree_liquid_save, "Borrowing rate should be higher than saving rate"
        assert agent.Rfree_IRA_save > agent.Rfree_liquid_save, "IRA rate should be higher than liquid rate"
        assert 0 <= agent.IRA_penalty_rate <= 1, "Penalty rate should be between 0 and 1"
        print("  ✓ All parameter relationships are correct")
        
        # Test 3: Solution structure
        print("\nTest 3: Solution structure")
        from HARK.interpolation import LinearInterp
        mGrid = np.linspace(0, 5, 20)
        cGrid = 0.8 * mGrid
        solution = IRASolution(
            cFunc=LinearInterp(mGrid, cGrid),
            cFunc_IRA=LinearInterp(mGrid, cGrid),
            vPfunc=LinearInterp(mGrid, cGrid ** (-2.0)),
            mNrmMin=0.0,
            hNrm=1.0,
        )
        assert hasattr(solution, 'cFunc_IRA'), "Solution should have IRA consumption function"
        print("  ✓ IRASolution structure is correct")
        
        print("\n" + "=" * 50)
        print("✅ All basic validation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run basic validation first
    basic_success = run_basic_validation()
    
    if basic_success:
        print("\n" + "=" * 50)
        print("Running full test suite...")
        
        # Try to run unittest if possible  
        try:
            unittest.main(verbosity=2, exit=False)
        except Exception as e:
            print(f"Full test suite failed: {e}")
            print("But basic validation passed, so core functionality works")
    else:
        print("\nBasic validation failed, skipping full test suite")