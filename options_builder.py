import math, numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class validated_input:
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
    S = validated_input(validator=validated_input.validate_positive)
    K = validated_input(validator=validated_input.validate_positive)
    T = validated_input(validator=validated_input.validate_positive_t)
    r = validated_input(validator=validated_input.validate_r)
    sigma = validated_input(validator=validated_input.validate_sigma)
    q = validated_input(validator=validated_input.validate_q)

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type.lower()  # Validate option_type

    def _d1(self):
        return (math.log(self.S / self.K) + (self.r - self.q + \
                0.5 * self.sigma**2) * self.T) / (self.sigma * math.sqrt(self.T))
    
    def _d2(self):
        return ((math.log(self.S / self.K) + (self.r - self.q + \
                0.5 * self.sigma**2) * self.T) / (self.sigma * math.sqrt(self.T)) - self.sigma * math.sqrt(self.T))
   
    def price(self):
        """
        Price an option using the Black-Scholes formula.
        """
        d1 = BlackScholes._d1(self)
        d2 = BlackScholes._d2(self)

        if self.option_type.lower() == "call":
            price = self.S * math.exp(-self.q * self.T) * norm.cdf(d1) - \
                self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option_type.lower() == "put":
            price = self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) - \
                self.S * math.exp(-self.q * self.T) * norm.cdf(-d1)
        return price   

    def delta(self):
        """
        Get the option delta.
        """
        if self.option_type.lower() == "call":
            return math.exp(-self.q * self.T) * norm.cdf(BlackScholes._d1(self))
        elif self.option_type.lower() == "put":
            return math.exp(-self.q * self.T) * (norm.cdf(BlackScholes._d1(self)) - 1)

    def gamma(self):
        """
        Get the option gamma.
        """
        return math.exp(-self.q * self.T) * norm.pdf(BlackScholes._d1(self)) / (self.S * \
                        self.sigma * math.sqrt(self.T))

    def vega(self):
        """
        Get the option vega.
        """
        return self.S * math.exp(-self.q * self.T) * norm.pdf(BlackScholes._d1(self)) * math.sqrt(self.T)

    def theta(self):
        """
        Get the option theta.
        """
        d1 = BlackScholes._d1(self)
        d2 = BlackScholes._d2(self)
        term1 = -(self.S * math.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma) / (2 * math.sqrt(self.T))
        if self.option_type.lower() == "call":
            term2 = self.q * self.S * math.exp(-self.q * self.T) * norm.cdf(d1)
            term3 = self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
            return term1 - term2 - term3
        elif self.option_type.lower() == "put":
            term2 = self.q * self.S * math.exp(-self.q * self.T) * norm.cdf(-d1)
            term3 = self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(-d2)
            return term1 + term2 + term3

    def rho(self):
        """
        Get the option rho.
        """
        if self.option_type.lower() == "call":
            return self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(BlackScholes._d2(self))
        elif self.option_type.lower() == "put":
            return -self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(-BlackScholes._d2(self))
        
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
                       "sigma": self.sigma, "q": self.q, "option_type": self.option_type}

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
                       "sigma": self.sigma, "q": self.q, "option_type": self.option_type}

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
