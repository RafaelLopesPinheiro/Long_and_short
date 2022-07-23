import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vectorbt as vbt
import scipy.stats as stats
plt.style.use('ggplot')


def get_data(tickers, startDate, endDate, interval):
    data = vbt.YFData.download(symbols=tickers, start=startDate, end=endDate, interval=interval).get('Close')
    return data


def create_csv(data):
    data.to_csv('petr_longshort.csv', index_label='Date')


def plot_reglin(x, y):
    plt.figure(figsize=(10,4))
    plt.plot(x.iloc[:, 0], x.iloc[:, 1],'ob', label='original data') 
    plt.plot(x.iloc[:, 0], y, '-r', label='fitted data')
    plt.legend()
    plt.title('Linear regression: %s x %s' % (x.columns[0], x.columns[1]), fontsize=15)
    plt.xlabel('{}'.format(x.columns[0]), size=14)
    plt.ylabel('{}'.format(x.columns[1]), size=14)

def z_score(resid):
    return resid - (resid.mean()/resid.std())


def delta_resid(resid, z_score):
    return z_score - z_score.shift(1)


def stats_t(slope, stderror):  ## STATS_T NEED TO BE MORE THAN (- 3.44) TO BE COINTEGRATED FOR SAMPLE SIZE > 100
    t = slope/stderror
    if t <= -3.44:  
        print('\nAssets Cointegrated!')
    else:
        print('Assets Not cointegrated!')
    print('Stats t =',t)
    return t


def plot_zscore(z):
    plt.figure(figsize=(10,4))
    plt.plot(z, '-k')
    plt.axhline(y= z.mean(), color='y')
    plt.axhline(y=z.mean() + (z.std()*2))
    plt.axhline(y=z.mean() - (z.std()*2))
    plt.xticks(rotation = -45, fontsize=15)
    plt.title('Z-Score', fontsize=20)
    plt.grid()
    


def half_life(z, reg_resid):
    half = round(-np.log10(2)/reg_resid.slope, 2)
    print('Half life =',half, 'dias. \n')
    



def main ():
    start_date = '2020-01-01 UTC'
    end_date = '2022-03-03 UTC'
    interval = '1D'
    tickers = ['PETR3.SA', 'PETR4.SA']
    
    ## LINES TO GET DATA FROM YAHOO FINANCE
    data = get_data(tickers, start_date, end_date, interval)
    create_csv(data)

    df = pd.read_csv('petr_longshort.csv', index_col='Date')
    df.index = pd.to_datetime(df.index)
    df.columns = [x.strip('.SA') for x in df.columns]  ## STRIP '.SA' FROM BRAZILIAN COMPANIES 


    ### MAKE THE COINTEGRATION WITH YOUR NUMBER OF DAYS ### 
    time_period = 300
    X_independent = df.iloc[-time_period:,0]
    Y_dependent = df.iloc[-time_period:,1]


    result = stats.linregress(X_independent, Y_dependent)
    y_pred = result.intercept + result.slope * X_independent
    residual = Y_dependent - y_pred


    ## PLOT DATA AND REGRESSION ## 
    plot_reglin(df[-time_period:], y_pred)


    z = z_score(residual)
    delta_res = delta_resid(residual, z)
    result2 = stats.linregress(x = z[:-1], y = delta_res[1:]) 


    stats_t(result2.slope, result2.stderr)
    print('Confidence level =',(100 * (1 - result2.pvalue))) 
    half_life(z, result2)
    plot_zscore(z)






if __name__ == '__main__':
    main()