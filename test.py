import numpy as np
import matplotlib.pyplot as plt
# Monte Carlo Simulation for Trading Strategy with 30% Win Rate, 2R Reward, and Stop-Out Limit
# Updated parameters
n_simulations = 10000  # Number of simulations
trades_per_year = 200  # Number of trades per year  
initial_balance = 12500  # Starting balance
risk_per_trade_pct = 0.01  # Risk per trade as a percentage of balance
win_rate = 0.30  # Win rate of the strategy
reward_r = 3  # average 2R reward
stop_out_limit = 0.3 * initial_balance  # 50% drawdown from starting balance
margin_trigger_balance = 30000  # Margin access trigger at $30k
margin_multiplier = 4  # Margin multiplier after $30k
risk_r = 1  # Risk per trade as a multiple of the risk percentage

# Run simulation with 2R reward and stop-out limit
simulations_30_win_margin_stop = []

for _ in range(n_simulations):
    balance = initial_balance
    balances = [balance]
    stopped_out = False
    for _ in range(trades_per_year):
        if stopped_out:
            balances.append(balance)
            continue

        # Determine effective risk percent based on margin access
        effective_risk_pct = risk_per_trade_pct
        if balance >= margin_trigger_balance:
            effective_risk_pct *= margin_multiplier

        # Simulate win/loss
        is_win = np.random.rand() < win_rate
        change_pct = reward_r * effective_risk_pct if is_win else -risk_r * effective_risk_pct
        balance *= (1 + change_pct)

        # Check stop-out condition
        if balance <= stop_out_limit:
            balance = stop_out_limit
            stopped_out = True

        balances.append(balance)
    simulations_30_win_margin_stop.append(balances)

# Plotting
plt.figure(figsize=(12, 6))
for sim in simulations_30_win_margin_stop:
    plt.plot(sim, alpha=0.5)
plt.title("Monte Carlo Simulation: 30% Win Rate, 2R, 4x Margin After $30k, Stop-Out at 50% Loss")
plt.xlabel("Trade Number")
plt.ylabel("Account Balance ($)")
plt.grid(True)
plt.axhline(initial_balance, color='black', linestyle='--', label='Starting Balance')
plt.axhline(margin_trigger_balance, color='blue', linestyle=':', label='Margin Trigger ($30k)')
plt.axhline(stop_out_limit, color='red', linestyle='--', label='Stop-Out Limit (50%)')
plt.legend()
plt.tight_layout()

# Return final balances for summary stats
final_balances_stop = [sim[-1] for sim in simulations_30_win_margin_stop]
final_balances_stop[:5], np.min(final_balances_stop), np.max(final_balances_stop), np.median(final_balances_stop)

print("Final Balances (first 5):", final_balances_stop[:5])
print("Minimum Final Balance:", np.min(final_balances_stop))
print("Maximum Final Balance:", np.max(final_balances_stop))
print("Median Final Balance:", np.median(final_balances_stop))
