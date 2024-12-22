
import platform
print("Current Python Version",platform.python_version())
if platform.python_version()<"3.11":
    print("ERROR: you are using a Python version lower than 3.11")
import numpy as np
from abc import ABC,abstractmethod
import time as time
import pandas as pd



#################################
######### PROBLEM 1 #############
#################################

class Backtester(ABC):
    def __init__(self, prices: np.array):
        """
        Initialize the Backtester with daily stock prices.
        
        Args:
            prices (2D np.array): A matrix where each row represents daily prices of stocks,
                                        and each column represents a stock.
        """
        self.prices = prices.copy() #deep copy of prices
        self.portfolio_values=np.zeros(prices.shape[0]) #portfolio value at each time step
        self.portfolio_shares=np.zeros((prices.shape[0],prices.shape[1])) # shares held in the portfolio for each time steps
        self.proportional_fees=0
    def display_portfolio(self):
        print("Portfolio values:",self.portfolio_values)
        print("Protfolio shares",self.portfolio_shares)



    def compute_fees(self,rebalance_period: int)->np.array:
        self.construct_equally_weighted_portfolio(rebalance_period)
        """
        compute fees incurred by rebalancing
        
        Output:
            array of daily fees
        """
        
        return 
    @abstractmethod
    def construct_equally_weighted_portfolio(self,rebalance_period: int):
        pass

class Nofees(Backtester):
    def __init__(self,prices):
       super().__init__(prices)
    def compute_rebalance_shares(self,portfolio_value: float,day_index: int)-> np.array:
        """
        Compute the numbers of shares for each stock to construct an equally weighted portfolio
        
        Args:
            portfolio_value (float): latest total value of the portfolio
            day_index (int): The index of the day for which we want to perform a rebalance
        
        Returns:
            np.array: A list where each element is number of shares needed to be held to construct an equally weighted porfolio
        """

        stock_nums = self.prices.shape[1]

        result = np.zeros(stock_nums)

        each_stock_cash = portfolio_value / stock_nums

        for i in range(stock_nums):
            shares = each_stock_cash / self.prices[day_index,i]
            result[i] = shares

        return result

    def construct_equally_weighted_portfolio(self, rebalance_period: int):
        """
        Implement an equally weighted portfolio with n-day rebalancing with an initial value of $1
        
        Args:
            rebalance_period (int): Number of days after which to rebalance the portfolio.
        
        Returns:
            
        """

        periods_len = self.prices.shape[0] # T+1

        self.portfolio_values[0] = 1000
        


        last_rebalance_date = 0

        for i in range(periods_len):
            if i % rebalance_period == 0:
                self.portfolio_values[i] = np.dot(self.prices[i,:], self.portfolio_shares[i-1,:]) if i > 0 else self.portfolio_values[0]
                self.portfolio_shares[i,:] = self.compute_rebalance_shares(self.portfolio_values[i],i)
                last_rebalance_date = i
            else:
                self.portfolio_shares[i,:] = self.portfolio_shares[last_rebalance_date,:]
                self.portfolio_values[i] = np.dot(self.prices[i,:], self.portfolio_shares[i,:])
        
            
class Proportionalfees(Backtester):
    def __init__(self,prices,p):
       super().__init__(prices)
       self.proportional_fees=p
    def compute_delta_change_in_share_value(self,shares:np.array,day_index: int)->np.array:
        """
        Computes the percentage change in amount of shares held 
        
        Args:
            shares_before (float): Number of shares held before rebalance
            shares_after (float): Number of shares held after rebalance
            day_index (int): The index of the day for which we want to perform a rebalance
        
        Returns:
            np.array: An array with the dollar value change on each stock
        """

        portfolio_temp = np.dot(shares,self.prices[day_index,:])
        delta_cash = np.zeros(self.prices.shape[1])

        for i in range(1,self.prices.shape[1]):
            delta_cash[i] = 1 / self.prices.shape[1] * (portfolio_temp - shares[i-1])
        return delta_cash
       


    def compute_target_shares(self,shares:np.array,day_index: int)->np.array:
        """
        Computes the target shares to obtain a equally balanced portfolio after proportional costs are deducted
        
        Args:
            shares_before (float): Number of shares held before rebalance
            shares_after (float): Number of shares held after rebalance
            day_index (int): The index of the day for which we want to perform a rebalance
        
        Returns:
            np.array: An array with the dollar value change on each stock
        """
       
        
      
    
    def construct_equally_weighted_portfolio(self, rebalance_period: int):
        """
        Add proportional fees to the backtest. Fees are applied during rebalancing.

        Args:
             rebalance_period (int): Number of days after which to rebalance the portfolio.

        Returns:
           
           
        """
       
       
    


#################################
######### PROBLEM 2 #############
#################################


#### Q1:

def generate_two_brownians(rho:float, T:float, nb_steps:int, nb_paths:int):
    """
        #Inputs:
        rho (float): correlation between the two Brownian motions
        T (float): time horizon
        nb_steps (int): number of time steps
        nb_paths (int): number of paths
        
        #Outputs:
        paths of B and W: np.array, np.array
    """

    delta_t = T / (nb_steps - 1)

    path_B = np.zeros([nb_steps,nb_paths])
    path_W = np.zeros([nb_steps,nb_paths])

    normal_matrix = np.random.normal(0,1,(nb_steps,nb_paths))
    normal_matrix2 = np.random.normal(0,1,(nb_steps,nb_paths))

    normal_matrix[0,:] = 0
    normal_matrix2[0,:] = 0

    path_W = np.cumsum(normal_matrix*np.sqrt(delta_t))
    path_W_ver = np.cumsum(normal_matrix*np.sqrt(delta_t))

    path_B = rho*path_B + np.sqrt(1-rho**2)*path_W_ver
    
    return path_B, path_W
    
    return     
#### Q2:

def generate_S_paths(params: np.array, nb_steps:int, nb_paths:int):
    """
        #Inputs:
        params = (S0, sigma0, nu, rho, t)
            - S0 (float): initial value of the stock price
            - sigma0 (float): initial value of the volatility
            - nu (float): volatility of volatility parameter
            - rho (float): correlation between the two Brownian motions
            - T (float): time horizon
        nb_steps (int): number of time steps
        nb_paths (int): number of paths
        
        #Outputs:
        paths of S: np.array

        
    """

    path_B, path_W = generate_two_brownians(params[3],params[4],nb_steps,nb_paths)

    sigma_matrix  = params[1]*np.exp(params[2]*path_W)

    path_S = params[0]*np.exp(sigma_matrix*path_B)
  
    
    return path_S


##### Q3:

def call_price_bergomi(K:float, params: np.array, nb_steps:int, nb_paths:int):
    """
        #Inputs:
        K (float): strike of the option
        params = (S0, sigma0, nu, rho, T)
            - S0 (float): initial value of the stock price
            - sigma0 (float): initial value of the volatility
            - nu (float): volatility of volatility parameter
            - rho (float): correlation between the two Brownian motions
            - T (float): time horizon
        nb_steps (int): number of time steps
        nb_paths (int): number of paths
        
        #Outputs:
        call Price: float
    """
    
    return 


##### Q4:

def barrier_price_bergomi(K:float, B:float, params: np.array, nb_steps:int, nb_paths:int):
    """
        #Inputs:
        K (float): strike of the option
        B (float): barrier of the option
        params = (S0, sigma0, nu, rho, T)
            - S0 (float): initial value of the stock price
            - sigma0 (float): initial value of the volatility
            - nu (float): volatility of volatility parameter
            - rho (float): correlation between the two Brownian motions
            - T (float): time horizon
        nb_steps (int): number of time steps
        nb_paths (int): number of paths
        
        #Outputs:
        barrier_price: float
        perc_valid: float
    """

    return 



#################################
######### PROBLEM 3 #############
#################################

#### Q1:

def calculate_moving_average(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    following specification
    Input:
    - df: A pandas DataFrame with dates as the index and stock names as columns. The values represent the stock's closing prices.
    - n: The number of days for the moving average (integer, e.g., 5 for a 5-day moving average).

    Output:
    - A pandas DataFrame with the same structure as the input but containing the n-day moving averages.

    """

    df_mv = 1 / n * sum(df.shift(i) for i in range(n))

    return df_mv
    

def detect_crossovers(df: pd.DataFrame, moving_averages: pd.DataFrame) -> pd.DataFrame:
    """
    following specification
    Input:
    - df: A pandas DataFrame with dates as the index and stock names as columns. The values represent the stock's closing prices.
    - moving_averages: A pandas DataFrame containing the n-day moving averages for each stock.

    Output:
    - A pandas DataFrame where each cell is True if a crossover occurs on that day, and False otherwise.

    """

    
    change = df - moving_averages

    result = ((change.shift(1) > 0) and (change < 0) ) or ((change.shift(1) < 0) and (change > 0) )
    
    return change
    
    
    
#### Q2:

def detect_anomalies(df: pd.DataFrame, z_score_threshold:float) -> pd.DataFrame:
    """
    following specification
    Input:
    - df: A pandas DataFrame with dates as the index and stock names as columns. The values represent the stock's closing prices.
    - z: The z-score threshold for anomaly flagging 

    Output:
    - A pandas DataFrame with the same structure that flags True for an anomaly and False for a "normal" data point

    """