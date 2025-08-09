"""
Example usage of the IRA Consumer Model.

This example demonstrates how to use the IRAConsumerType to model
an agent with both liquid and IRA accounts, including early withdrawal penalties.
"""

import sys
sys.path.insert(0, '/home/runner/work/HARK/HARK')

# This would normally import the IRA model
# from HARK.ConsumptionSaving.ConsIRAModel import IRAConsumerType, init_ira_accounts

def example_ira_usage():
    """
    Example of how to use the IRA Consumer Model.
    
    This function shows the typical workflow for:
    1. Setting up an agent with IRA accounts
    2. Configuring early withdrawal penalties
    3. Solving the model
    4. Running simulations
    """
    
    print("IRA Consumer Model Example")
    print("=" * 40)
    
    # Step 1: Set up parameters for IRA model
    print("\n1. Setting up IRA model parameters:")
    
    ira_params = {
        # Standard consumption model parameters
        'cycles': 0,                    # Infinite horizon
        'T_cycle': 1,                   # Single period type
        'CRRA': 2.0,                    # Risk aversion
        'DiscFac': 0.96,                # Discount factor
        'LivPrb': [0.98],               # Survival probability
        'PermGroFac': [1.01],           # Income growth
        
        # IRA-specific parameters
        'Rfree_liquid_save': 1.03,      # 3% on liquid savings
        'Rfree_liquid_boro': 1.20,      # 20% on liquid borrowing
        'Rfree_IRA_save': 1.07,         # 7% on IRA savings (higher return)
        'Rfree_IRA_boro': 1.00,         # No borrowing from IRA allowed
        'IRA_penalty_rate': 0.10,       # 10% early withdrawal penalty
        'retirement_age': 65,           # Penalty-free age
        
        # Simulation parameters
        'AgentCount': 10000,            # Number of agents to simulate
        'T_sim': 200,                   # Simulation periods
        'T_age': 100,                   # Maximum age
    }
    
    print(f"  - Liquid account saving rate: {ira_params['Rfree_liquid_save']:.1%}")
    print(f"  - IRA account saving rate: {ira_params['Rfree_IRA_save']:.1%}")
    print(f"  - Early withdrawal penalty: {ira_params['IRA_penalty_rate']:.1%}")
    print(f"  - Retirement age: {ira_params['retirement_age']}")
    
    # Step 2: Create IRA consumer agent
    print("\n2. Creating IRA consumer agent:")
    print("  agent = IRAConsumerType(**ira_params)")
    
    # Step 3: Solve the model
    print("\n3. Solving the consumption-saving problem:")
    print("  agent.solve()")
    print("  # This solves for optimal consumption and saving in both accounts")
    print("  # considering the early withdrawal penalty")
    
    # Step 4: Examine policy functions
    print("\n4. Examining policy functions:")
    print("  # Consumption function over liquid assets:")
    print("  cFunc_liquid = agent.solution[0].cFunc")
    print("  # Consumption function over IRA assets:")  
    print("  cFunc_IRA = agent.solution[0].cFunc_IRA")
    
    # Step 5: Run simulation
    print("\n5. Running life-cycle simulation:")
    print("  agent.initialize_sim()")
    print("  agent.simulate()")
    print("  # Simulates agents' choices over their lifetime")
    
    # Step 6: Analyze results
    print("\n6. Analyzing results:")
    print("  # Average liquid assets by age:")
    print("  liquid_assets = agent.history['aNrm']")
    print("  # Average consumption by age:")
    print("  consumption = agent.history['cNrm']")
    print("  # Fraction using IRA vs liquid savings by age")
    
    print("\n" + "=" * 40)
    print("Key Features of the IRA Model:")
    print("- Two savings accounts: liquid and IRA")
    print("- Kinked interest rates for each account") 
    print("- Early withdrawal penalties for IRA")
    print("- Optimal allocation between accounts")
    print("- Age-dependent penalty structure")


def compare_scenarios():
    """
    Compare different IRA scenarios to show model capabilities.
    """
    
    print("\nScenario Comparison")
    print("=" * 40)
    
    scenarios = {
        'No IRA': {
            'description': 'Traditional single liquid account',
            'Rfree_liquid_save': 1.03,
            'Rfree_IRA_save': 1.03,     # Same as liquid
            'IRA_penalty_rate': 1.0,    # 100% penalty = never use IRA
        },
        
        'High Penalty IRA': {
            'description': 'IRA with 20% early withdrawal penalty',
            'Rfree_liquid_save': 1.03,
            'Rfree_IRA_save': 1.08,     # Higher return
            'IRA_penalty_rate': 0.20,   # 20% penalty
        },
        
        'Low Penalty IRA': {
            'description': 'IRA with 5% early withdrawal penalty',
            'Rfree_liquid_save': 1.03,
            'Rfree_IRA_save': 1.08,     # Higher return  
            'IRA_penalty_rate': 0.05,   # 5% penalty
        },
        
        'No Penalty After 59.5': {
            'description': 'Realistic IRA with age-based penalties',
            'Rfree_liquid_save': 1.03,
            'Rfree_IRA_save': 1.08,     # Higher return
            'IRA_penalty_rate': 0.10,   # 10% penalty before 59.5
            'retirement_age': 59.5,     # Penalty ends at 59.5
        }
    }
    
    print("Expected behaviors:")
    for name, params in scenarios.items():
        print(f"\n{name}:")
        print(f"  {params['description']}")
        print(f"  - Young agents: {'Prefer liquid' if params['IRA_penalty_rate'] > 0.15 else 'Use both accounts'}")
        print(f"  - Older agents: {'Shift to IRA' if params.get('retirement_age', 65) < 65 else 'Continue mixed strategy'}")


if __name__ == "__main__":
    example_ira_usage()
    compare_scenarios()
    
    print("\n" + "=" * 50)
    print("NOTE: This is a demonstration of the IRA model structure.")
    print("To run actual simulations, install required dependencies:")
    print("  pip install numpy scipy matplotlib pandas")
    print("Then import and use the IRAConsumerType class.")