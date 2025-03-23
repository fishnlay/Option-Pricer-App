import math, numpy as np, pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class _validated_input:
    def __init__(self, name=None, validator=None):
        self.name = name
        self.validator = validator

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, type=None) -> object:
        return obj.__dict__.get(self.name, 0)

    def __set__(self, obj, value) -> None:
        if self.validator:
            self.validator(value, self.name)
        obj.__dict__[self.name] = value

    @staticmethod
    def validate_positive(value, name):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a positive int or float.")

    @staticmethod
    def validate_positive_t(value, name):
        if not isinstance(value, (int, float)) or value <= 1 / 10000: #must be some time left before expiration
            raise ValueError(f"{name} must be a positive int or float greater than 1/10000.")

    @staticmethod
    def validate_r(value, name):
        """
        Input validation for r with 0.1 being 10%.
        This ensure the user is using the correct scale.
        """
        if not isinstance(value, (int, float)) or not (-1 <= value <= 1): #as of 100 and -100%
            raise ValueError(f"{name} must be an int or float between -1 and 1.")

    @staticmethod
    def validate_sigma(value, name):
        if not isinstance(value, (int, float)) or not (0 < value <= 4): #define upper bounds for volatility
            raise ValueError(f"{name} must be a positive int or float no greater than 4.")

    @staticmethod
    def validate_q(value, name):
        if not isinstance(value, (int, float)) or not (0 <= value <= 1):
            raise ValueError(f"{name} must be a positive int or float between 0 and 1.")

class BlackScholes:
    """
    Black-Scholes option pricing model for European Vanilla Options.

    :S: Spot price
    :K: Strike price
    :T: Time in years
    :r: risk-free rate -> .04 being 4%
    :sigma: volatility -> .30 being 30%
    :q: dividend yield ->.05 being 5%
    """
    S = _validated_input(validator=_validated_input.validate_positive)
    K = _validated_input(validator=_validated_input.validate_positive)
    T = _validated_input(validator=_validated_input.validate_positive_t)
    r = _validated_input(validator=_validated_input.validate_r)
    sigma = _validated_input(validator=_validated_input.validate_sigma)
    q = _validated_input(validator=_validated_input.validate_q)
    quantity = _validated_input(validator=_validated_input.validate_positive)

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str, position='long', quantity:int=1):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type.lower()  # Validate option_type
        self.position = position.lower() if position.lower() in ['long', 'short'] else ValueError("Position must be 'long' or 'short' ")
        self.quantity = quantity

    def _d1(self):
        return (math.log(self.S / self.K) + (self.r - self.q + \
                0.5 * self.sigma**2) * self.T) / (self.sigma * math.sqrt(self.T))
    
    def _d2(self):
        return ((math.log(self.S / self.K) + (self.r - self.q + \
                0.5 * self.sigma**2) * self.T) / (self.sigma * math.sqrt(self.T)) - self.sigma * math.sqrt(self.T))
   
    def price(self):
        """Price an option using the Black-Scholes formula."""
        d1 = BlackScholes._d1(self)
        d2 = BlackScholes._d2(self)

        if self.option_type.lower() == "call":
            price = self.S * math.exp(-self.q * self.T) * norm.cdf(d1) - \
                self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option_type.lower() == "put":
            price = self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) - \
                self.S * math.exp(-self.q * self.T) * norm.cdf(-d1)
        return price * self.quantity

    def delta(self):
        """Get the option delta."""
        sign = 1 if self.position == "long" else -1

        if self.option_type.lower() == "call":
            return math.exp(-self.q * self.T) * norm.cdf(BlackScholes._d1(self)) * self.quantity * sign
        elif self.option_type.lower() == "put":
            return math.exp(-self.q * self.T) * (norm.cdf(BlackScholes._d1(self)) - 1) * self.quantity * sign

    def gamma(self):
        """Get the option gamma."""
        sign = 1 if self.position == "long" else -1

        return math.exp(-self.q * self.T) * norm.pdf(BlackScholes._d1(self)) / (self.S * \
                        self.sigma * math.sqrt(self.T)) * self.quantity * sign

    def vega(self):
        """Get the option vega."""
        sign = 1 if self.position == "long" else -1

        return self.S * math.exp(-self.q * self.T) * norm.pdf(BlackScholes._d1(self)) * math.sqrt(self.T) * self.quantity * sign

    def theta(self):
        """Get the option theta."""
        sign = 1 if self.position == "long" else -1

        d1 = BlackScholes._d1(self)
        d2 = BlackScholes._d2(self)
        term1 = -(self.S * math.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma) / (2 * math.sqrt(self.T))
        if self.option_type.lower() == "call":
            term2 = self.q * self.S * math.exp(-self.q * self.T) * norm.cdf(d1)
            term3 = self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
            return (term1 - term2 - term3) * self.quantity * sign
        elif self.option_type.lower() == "put":
            term2 = self.q * self.S * math.exp(-self.q * self.T) * norm.cdf(-d1)
            term3 = self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(-d2)
            return (term1 + term2 + term3)* self.quantity * sign

    def rho(self):
        """Get the option rho."""
        sign = 1 if self.position == "long" else -1

        if self.option_type.lower() == "call":
            return self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(BlackScholes._d2(self)) * self.quantity * sign
        elif self.option_type.lower() == "put":
            return -self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(-BlackScholes._d2(self)) * self.quantity * sign
        
    def expiry_profits(self, spot):
        """Calculate the profits at expiry for the option."""
        base_params = {"S": self.S, "K": self.K, "T": self.T, "r": self.r, 
                       "sigma": self.sigma, "q": self.q, "option_type": self.option_type, 
                       "position": self.position, "quantity": self.quantity}
        
        option_price = BlackScholes(**base_params).price()

        if self.option_type not in ('call', 'put'):
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        sign = 1 if self.position == "long" else -1
        if self.option_type == 'call':
            return sign * (max(0, spot - self.K) - option_price) * self.quantity
        elif self.option_type == 'put':
            return sign * (max(0, self.K - spot) + option_price) * self.quantity
        
    def today_profits(self, spot):
        """Calculate the profits at time 0 for the option."""
        base_params = {"S": self.S, "K": self.K, "T": self.T, "r": self.r, 
                       "sigma": self.sigma, "q": self.q, "option_type": self.option_type, 
                       "position": self.position, "quantity": self.quantity}
        
        buy_price = BlackScholes(**base_params).price()

        base_params = {"S": spot, "K": self.K, "T": self.T, "r": self.r, 
                       "sigma": self.sigma, "q": self.q, "option_type": self.option_type, 
                       "position": self.position, "quantity": self.quantity}
        
        sell_price = BlackScholes(**base_params).price()
        return (sell_price- buy_price) if self.position == "long" else (buy_price - sell_price)

    #PLOTS    
    def heatmap(self, param1, range1, param2, range2, figsize=(10,10)):
        """
        Generate a heatmap by varying two parameters and computing option prices.

        :param param1: The name of the first parameter to vary (y axis)
        :param param2: The name of the second parameter to vary (x axis)
        """

        #Initialize a table
        param1_values = np.arange(range1[0], range1[1], (range1[1] - range1[0])/10.0)
        param2_values = np.arange(range2[0], range2[1], (range2[1] - range2[0])/10.0)
        heatmap = np.zeros((len(param1_values), len(param2_values)))

        base_params = {"S": self.S, "K": self.K, "T": self.T, "r": self.r, 
                       "sigma": self.sigma, "q": self.q, "option_type": self.option_type, "position": self.position}

        for i, param1_value in enumerate(param1_values):
            for j, param2_value in enumerate(param2_values):
                base_params[param1] = param1_value
                base_params[param2] = param2_value
                option_price = BlackScholes(**base_params).price()
                heatmap[i, j] = option_price

        if param1 == "T":
            param1_values = param1_values*252
        elif param2 == "T":
            param2_values = param2_values*252

        
        #Color map
        sample = 20
        colors = np.vstack(
            (# Default is White -> Green but I want Green -> White -> Red so reverse the order
                plt.get_cmap("Reds", sample)(np.linspace(0, 1, sample))[::-1],
                # np.ones((1, 4)),  # Explicit White optional
                plt.get_cmap("Greens", sample)(np.linspace(0, 1, sample)),
            ))
            
        cm = LinearSegmentedColormap.from_list("green_to_red", colors)
        
        sns.set_theme(rc={'figure.figsize':figsize})
        ax = sns.heatmap(heatmap, annot=True, fmt=".2f", cmap=cm)
        ax.set_xlabel(param2)
        ax.set_ylabel(param1)

        #Display the ticks
        ax.set_xticks(np.arange(len(param2_values)) + 0.5)  
        ax.set_yticks(np.arange(len(param1_values)) + 0.5)
        ax.set_xticklabels([f"{val:.2f}" for val in param2_values], rotation=45)
        ax.set_yticklabels([f"{val:.2f}" for val in param1_values], rotation=0)
        
    def greek_sensitivity(self, greek:str):
        """
        Collect data fr delta sensitivity to strike prices

        :greek: greek choosen for the sensitivity test
        """
        min_S, max_S, step_S = self.S * 0.001, self.S * 2.5, self.S * 0.01

        #Initialize 
        strikes = np.arange(min_S, max_S, step_S)
        greek_values  = {}

        base_params = {"S": self.S, "K": self.K, "T": self.T, "r": self.r, 
                       "sigma": self.sigma, "q": self.q, "option_type": self.option_type, "position": self.position}

        for _, strike in enumerate(strikes):
            base_params['S'] = strike

            bs_model = BlackScholes(**base_params)
            if hasattr(bs_model, greek.lower()): 
                greek_func = getattr(bs_model, greek.lower())
                greek_values[strike] = greek_func()
            else:
                raise ValueError(f"Invalid Greek name: '{greek.lower()}'")
        return greek_values 
    
    def greek_lineplot(self, greek:str, figsize=(10,7)):
        """
        Plot Greek sensitivity to the underlying stock price.

        :greek: greek choosen for the sensitivity test
        :figsize: tuple for the figure size
        """
        sns.set_theme(rc={'figure.figsize':figsize})
        sns.lineplot(self.greek_sensitivity(greek))

class PositionBuilder: 
    def __init__(self):
        self.options = []

    def add_option(self, option):
        """Add an Option object to the position."""
        if isinstance(option, list):
            self.options.extend(option)
        else:
            self.options.append(option)

    def remove_option(self, index):
        """Remove an option by its index."""
        if 0 <= index < len(self.options):
            self.options.pop(index)
    
    def total_expiry_profits(self, spot):
        """Calculate the total profits for a given spot price."""
        return sum(option.expiry_profits(spot) for option in self.options)
    
    def total_today_profits(self, spot):
        """Calculate the total profits for a given spot price."""
        return sum(option.today_profits(spot) for option in self.options)
    
    def shift_parameter(self, param, new_val):
        """"Shift a parameter (Time, Vol or Rate) for all options in the portfolio"""

        for option in self.options:
            if param == 'T':
                option.T -= new_val/252.0
            elif param == 'sigma':
                option.sigma *= 1+(new_val/100.0)
            elif param == 'r':
                option.r *= 1+(new_val/100.0)
            else:
                raise ValueError(f"{param} must be 'T', 'sigma' or 'r' ")

    def positions(self, S):
        """Return a DataFrame with PnL for today and at expiry."""
        min_S, max_S, step_S = S * 0.1, S * 2.0, S * 0.01
        spots = np.arange(min_S, max_S, step_S)

        today_pnl = np.array([self.total_today_profits(spot) for spot in spots])
        expiry_pnl = np.array([self.total_expiry_profits(spot) for spot in spots])

        df = pd.DataFrame({
            'Spot': spots,
            'PnL Today': today_pnl,
            'PnL at Expiry': expiry_pnl
        }).set_index('Spot')
        return df
    
    def pos_table(self):
        """Return a table with all the position in the book and their greeks..."""

        data = {"Amount": [],
                "Position": [],
                "Type": [],
                "Expiry Date": [],
                "Strike": [],
                "Price": [],
                "Delta": [],
                "Gamma": [],
                "Theta": [],
                "Vega": [],
                "Rho": []}
        
        for option in self.options:
            data['Amount'].append(option.quantity)
            data['Position'].append(option.position.capitalize())
            data['Type'].append(option.option_type.capitalize())
            data['Expiry Date'].append(option.T * 252)
            data['Price'].append(option.price())
            data['Strike'].append(option.K)
            data['Delta'].append(option.delta())
            data['Gamma'].append(option.gamma())
            data['Theta'].append(option.theta())
            data['Vega'].append(option.vega())
            data['Rho'].append(option.rho())

        return data
