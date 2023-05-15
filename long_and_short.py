import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.stats as stats
import os
import datetime as dt


def download_data(tickers, startDate, endDate, interval):
    """Download OHLC data of the tickers

    Args:
        tickers (str or list): tickers of the assets
        startDate (str): start date period
        endDate (str): end date period
        interval (str): interval types: 1D, 1H, 1M

    Returns:
        pd.DataFrame: data
    """
    data = yf.download(tickers=tickers, start=startDate, end=endDate, interval=interval)['Adj Close']
    data.ffill(inplace=True)
    return data


def create_csv(data, file_name):
    data.to_csv(file_name)


def load_data(tickers, startDate, endDate, interval):
    """Create a csv with the stocks data for saving and logging purpose.

    Args:
        tickers (str or list): tickers of the assets
        startDate (str): start date period
        endDate (str): end date period
        interval (str): interval types: 1D, 1H, 1M

    Returns:
        pd.DataFrame : DataFrame created
    """
    file_name = f"{tickers[0]}_{tickers[1]}.csv"
    data = download_data(tickers, startDate, endDate, interval)
    create_csv(data, file_name)
    return pd.read_csv(file_name, index_col='Date')


def plot_reglin(x, y):
    """Plot the linear regression model for the X and Y variables.

    Args:
        x (pd.DataFrame): The exogenous variabels of the model
        y (pd.DataFrame): The Endogenous variables of the model
    """
    plt.figure(figsize=(10,4))
    plt.plot(x.iloc[:, 0], x.iloc[:, 1],'ob', label='original data') 
    plt.plot(x.iloc[:, 0], y, '-r', label='fitted data')
    plt.legend()
    plt.title('Linear regression: %s x %s' % (x.columns[0], x.columns[1]), fontsize=15)
    plt.xlabel('{}'.format(x.columns[0]), size=14)
    plt.ylabel('{}'.format(x.columns[1]), size=14)


def z_score(resid):
    """Calculate the Z score by subtracting the residual difference between the mean and standard deviation ratio.
    
    Args:
        resid (pd.Series): Residual between Y_true and Y_pred.

    Returns:
        float: Z score value.
    """
    return resid - (resid.mean()/resid.std())


def delta_resid(z_score):
    return z_score - z_score.shift(1)


def stats_t(slope, stderror, cut_off):  ## STATS_T NEED TO BE MORE THAN (- 3.44) TO BE COINTEGRATED FOR SAMPLE SIZE > 100
    """Calculate T-statistic and check if is lower than cut_off value to be cointegrated for sample size > 100

    Args:
        slope (float): Slope of the regression
        stderror (float): Stdev of regression
        cut_off (float): Cut off value for cointegration

    Returns:
        float : Stats-T value
    """
    t = slope/stderror
    if t <= cut_off:  
        print('\nAssets Cointegrated!')
    else:
        print('Assets Not cointegrated!')
    print(f'Stats t = {round(t,5)}')
    return t


def plot_zscore(z, names):
    plt.figure(figsize=(10,4))
    plt.plot(z, '-k')
    plt.axhline(y= z.mean(), color='y')
    plt.axhline(y=z.mean() + (z.std()*2))
    plt.axhline(y=z.mean() - (z.std()*2))
    plt.xticks(rotation = -45, fontsize=15)
    plt.title(f'Z-Score ({names[0]} x {names[1]})', fontsize=20)
    plt.text(z.index[0], (z.mean() + (z.std()*2))*1.10, 'If Z_score >= upper line, sell Y and buy X')
    plt.text(z.index[0], (z.mean() - (z.std()*2))*1.13, 'If Z_score <= bottom line, buy Y and sell X')
    plt.grid()
    plt.show()

    

def half_life(z, reg_resid):
    """Calculate half life of the cointegration

    Args:
        z (float): Z-Score 
        reg_resid (list): Z-Score regression results
    """
    half = round(-np.log10(2)/reg_resid.slope, 2) ## Half life formula
    print('Half life =',half, 'dias. \n')
    

def size_position(data, slope):
    """Print the ratio of each stock to buy/sell

    Args:
        data (pd.DataFrame): _description_
        slope (float): _description_
    """
    print(f'Ratio = {slope*100:.0f} shares of', f'{data.columns[0]}', f'\n \tper 100 shares of {data.columns[1]}')



def main ():
    start_date = dt.date(2020, 1, 1)
    end_date = dt.date(2022, 7, 3)
    interval = '1D'
    tickers = [str(input('Enter the ticker name (Yahoo Finance): ')) for i in range(2)]

    df = load_data(tickers, start_date, end_date, interval)
    df.index = pd.to_datetime(df.index)
    if "(.SA)" in df.columns:
        df.columns = [x.strip('(.SA)') for x in df.columns]  ## STRIP '.SA' FROM BRAZILIAN COMPANIES


    ## MAKE THE COINTEGRATION WITH YOUR NUMBER OF DAYS ## 
    time_period = 500
    X_independent = df.iloc[-time_period:,0]
    Y_dependent = df.iloc[-time_period:,1]


    result = stats.linregress(X_independent, Y_dependent)
    y_pred = result.intercept + result.slope * X_independent
    residual = Y_dependent - y_pred


    ## PLOT DATA AND REGRESSION ## 
    plot_reglin(df[-time_period:], y_pred)


    z = z_score(residual)
    delta_res = delta_resid(z)
    z_regression = stats.linregress(x = z[:-1], y = delta_res[1:]) 

    cut_off = -3.43
    stats_t(z_regression.slope, z_regression.stderr, cut_off)
    print('Confidence level =',(round(100 * (1 - z_regression.pvalue), 5))) 
    half_life(z, z_regression)
    plot_zscore(z, df.columns.values)

    size_position(df, result.slope)


if __name__ == '__main__':
    main()