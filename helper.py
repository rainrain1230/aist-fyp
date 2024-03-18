import numpy as np

def sortino_ratio(portfolio_returns, risk_free_rate=0):
    expected_return = np.mean(portfolio_returns)
    
    downside_returns = np.minimum(0, portfolio_returns - risk_free_rate)
    
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_deviation == 0:
        return 0
    
    sortino_ratio = (expected_return - risk_free_rate) / downside_deviation
    
    return sortino_ratio