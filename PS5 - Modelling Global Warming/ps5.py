# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: Amaan Karim Ahmad 

import pylab
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

"""
End helper code
"""

def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    # Initialise list of pylab arrays
    coefficients = []
    for poly in degs:
        coefficients.append(pylab.polyfit(x,y,poly))
    
    return coefficients


def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    # Calculate mean of y values
    mean = pylab.mean(y)
    # Calculate the sum of the squared differences between y and estimated
    sum_squared_diff_est = sum((y-estimated)**2)
    # Calculate the sum of the squared differences between the y and its mean
    sum_squared_diff_mean = sum((y-mean)**2)
    return (1 - (sum_squared_diff_est/sum_squared_diff_mean))


def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    # Iterating through list of regression models
    for model in models:
        # Plot labels
        pylab.xlabel("Time (Years)")
        pylab.ylabel("Temperature (Degrees Celcius)")
        # Plot Observed Data with blue circles
        pylab.plot(x,y,'bo', label = 'Observed Data')
        # Obtain the model y values
        estimated = pylab.polyval(model, x)
        # Plot the Model Data
        pylab.plot(x,estimated, 'r-', label = "Model Data")

        # Find the number of degrees for this model
        degree = len(model) - 1
        # Obtain the R squared value for this model
        R_squared = r_squared(y, estimated)

        # If the Model is linear
        if degree == 1:
            standard_error = se_over_slope(x, y, estimated, model)
            title = 'Climate Regression, Model of {0} Degree\nR-squared: {1:3f}, SE/slope: {2:.3f}'.format(degree, R_squared, standard_error)
        else: 
            title = 'Climate Regression, Model of {0} Degrees\nR-squared: {1:3f}'.format(degree, R_squared)

        # Output graphs
        pylab.title(title)
        pylab.show()

def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    # Initialise pylab array of average annual temperature over the given cities
    avg_annual_national_temp = pylab.array([])

    for year in years:
        avg_multicity_temperatures = []
        for city in multi_cities:
            avg_city_temperature = climate.get_yearly_temp(city, year).mean()
            avg_multicity_temperatures.append(avg_city_temperature)
        avg_national_temp = pylab.array(avg_multicity_temperatures).mean()
        avg_annual_national_temp = pylab.append(avg_annual_national_temp, avg_national_temp)

    return avg_annual_national_temp


def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    # Initialise pylab array of moving averages
    moving_avg = []

    # For 
    for i in range(len(y)):
        if i < window_length:
            moving_avg.append(sum(y[0 : i+1])/(i + 1))
        else:
            moving_avg.append(sum(y[(i+1 - window_length) : i+1])/window_length)
    
    return pylab.array(moving_avg)

def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    # Calculate the sum of the squared differences between y and estimated
    sum_squared_diff = sum((y-estimated)**2)
    return (float(pylab.sqrt(sum_squared_diff/len(y))))


def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    # Initialise multi_city annual standard deviations
    annual_multi_cities_std = []
    # Iterate through the years
    for year in years:
        daily_avgs = []
        for month in range(1,13):
            for day in range(1,32):
                try:
                    city_day_data = []
                    for city in multi_cities:
                        city_day_data.append(climate.get_daily_temp(city, month, day,year))
                    daily_avg = pylab.array(city_day_data).mean()
                    daily_avgs.append(daily_avg)
                except AssertionError:
                    pass
        annual_multi_cities_std.append(pylab.std(daily_avgs))
    return pylab.array(annual_multi_cities_std)

def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    # Iterating through list of regression models
    for model in models:
        # Plot labels
        pylab.xlabel("Time (Years)")
        pylab.ylabel("Temperature (Degrees Celcius)")
        # Plot Observed Data with blue circles
        pylab.plot(x,y,'bo', label = 'Observed Data')
        # Obtain the model y values
        estimated = pylab.polyval(model, x)
        # Plot the Model Data
        pylab.plot(x,estimated, 'r-', label = "Model Data")

        # Find the number of degrees for this model
        degree = len(model) - 1
        # Obtain the R squared value for this model
        RMSE = rmse(y, estimated)

        title = 'Climate Model Prediction\nModel of {0} Degree(s), RMSE: {1:3f}'.format(degree, RMSE)
        # Output graphs
        pylab.title(title)
        pylab.show()


if __name__ == '__main__':

    # Obtain the data
    data = Climate("data.csv")
    years = pylab.array(TRAINING_INTERVAL)
    testing_years = pylab.array(TESTING_INTERVAL)

    # Part A.4.I
    print("Starting Jan 10th Test")
    jan10 = pylab.array([data.get_daily_temp('NEW YORK', 1, 10, year) for year in TRAINING_INTERVAL])
    modelA = generate_models(years, jan10, [1])
    evaluate_models_on_training(years, jan10, modelA)
    
    # Part A.4.II
    print("Starting Annual Test")
    annual = pylab.array([data.get_yearly_temp('NEW YORK', year).mean() for year in TRAINING_INTERVAL])
    modelA2 = generate_models(years, annual, [1])
    evaluate_models_on_training(years, annual, modelA2)

    # Part B
    print("Starting National Annual Test")
    national_annual = gen_cities_avg(data, CITIES, TRAINING_INTERVAL)
    modelB = generate_models(years, national_annual, [1])
    evaluate_models_on_training(years, national_annual, modelB)

    # Part C
    print("Starting Moving Average Test")
    moving_averages = moving_average(gen_cities_avg(data, CITIES, TRAINING_INTERVAL), 5)
    modelC = generate_models(years, moving_averages, [1])
    evaluate_models_on_training(years, national_annual, modelC)

    # Part D.2
    print("Starting Prediction Test")
    print("Training more models...")
    moving_averages_prediction = moving_average(gen_cities_avg(data, CITIES, TRAINING_INTERVAL), 5)
    modelD = generate_models(years, moving_averages_prediction, [1,2,20])
    evaluate_models_on_training(years, moving_averages_prediction, modelD)
    print("Predicting from models...")
    testing_avg = gen_cities_avg(data, CITIES, testing_years) 
    evaluate_models_on_testing(testing_years, testing_avg, modelD)

    # Part E
    print("Starting Extreme Test")
    moving_averages_prediction_std = moving_average(gen_std_devs(data, CITIES, TRAINING_INTERVAL), 5)
    modelE = generate_models(years, moving_averages_prediction_std, [1])
    evaluate_models_on_training(years, moving_averages_prediction_std, modelE)