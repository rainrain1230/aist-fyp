import pandas as pd
from enhanced_neat import StatisticsReporter, StdOutReporter
from enhanced_neat.nn import FeedForwardNetwork
from enhanced_neat.config import Config
from enhanced_neat.population import Population
from enhanced_neat.genome import DefaultGenome
from enhanced_neat.reproduction import DefaultReproduction
from enhanced_neat.species import DefaultSpeciesSet
from enhanced_neat.stagnation import DefaultStagnation
import numpy as np
import pandas_ta as ta
import random
from sklearn.preprocessing import MinMaxScaler
from helper import sortino_ratio

random.seed(9001)
training_window_size = 100
testing_windows_size = 20

stocks = ['AAPL']
result_all = []

for stock in stocks:

    print(stock)

    data = pd.read_csv(f'preprocessed_data/{stock}_preprocessed_data.csv')


    long_capital = 10000
    short_capital = 10000

    config = Config(DefaultGenome, DefaultReproduction,
                        DefaultSpeciesSet, DefaultStagnation,
                        'config_neat.ini')

    config_short = Config(DefaultGenome, DefaultReproduction,
                        DefaultSpeciesSet, DefaultStagnation,
                        'config_neat_short.ini')
    
    num_long = 0
    num_short = 0
    holding_long = False
    holding_short = False
    buy_price = 0
    long_win_num = 0
    short_win_num = 0

    trade_records = []

    for i in range(0, len(data) - training_window_size-testing_windows_size, testing_windows_size):
        train_data = data.iloc[i:i+training_window_size].copy()
        test_data = data.iloc[i+training_window_size+1:i+training_window_size+1+testing_windows_size].copy()

        train_data['daily_return_minmax'] = MinMaxScaler().fit_transform(train_data['daily_return'].values.reshape(-1, 1))
        test_data['daily_return_minmax'] = MinMaxScaler().fit_transform(test_data['daily_return'].values.reshape(-1, 1))

        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                net = FeedForwardNetwork.create(genome, config)
                holding_stock = False
                buy_price = 0
                portfolio_returns = []

                for j in range(6, training_window_size):

                    thresholds = [int(x*100) for x in net.activate((np.array(train_data['daily_return_minmax'].iloc[j-5:j])*100))]
                    
                    if not (5 <= thresholds[0] <= 95 and thresholds[0] < thresholds[1] <= 95):
                        genome.fitness = 0
                    else:
                        buy_signal = (train_data['rsi'].iloc[j] < thresholds[0]) * 1
                        sell_signal = (train_data['rsi'].iloc[j] > thresholds[1]) * -1

                        if train_data['adjClose'].iloc[j] > train_data['ema'].iloc[j]:
                            if buy_signal == 1 and not holding_stock:
                                buy_price = train_data['adjClose'].iloc[j]
                                holding_stock = True
                            elif sell_signal == -1 and holding_stock:
                                sell_price = train_data['adjClose'].iloc[j]
                                portfolio_returns.append(sell_price / buy_price - 1)
                                holding_stock = False

                if holding_stock == True:
                    sell_price = train_data['adjClose'].iloc[-1]
                    profit = sell_price / buy_price - 1
                    portfolio_returns.append(profit)

                genome.fitness = np.mean(portfolio_returns)

        def eval_genomes_short(genomes, config):
            for genome_id, genome in genomes:
                net = FeedForwardNetwork.create(genome, config)
                holding_stock = False
                buy_price = 0
                portfolio_returns = []

                for j in range(6, training_window_size):
                    thresholds = [int(x*100) for x in net.activate((np.array(train_data['daily_return_minmax'].iloc[j-5:j])*100))]
                    if not (5 <= thresholds[0] <= 95 and thresholds[0] < thresholds[1] <= 95):
                        genome.fitness = 0
                    else:
                        buy_signal = (train_data['rsi'].iloc[j] < thresholds[0]) * 1
                        sell_signal = (train_data['rsi'].iloc[j] > thresholds[1]) * -1

                        if train_data['adjClose'].iloc[j] < train_data['ema'].iloc[j]:
                            if sell_signal == -1 and not holding_stock:
                                sell_price = train_data['adjClose'].iloc[j]
                                holding_stock = True
                            elif buy_signal == 1 and holding_stock:
                                buy_price = train_data['adjClose'].iloc[j]
                                portfolio_returns.append(sell_price / buy_price - 1)
                                holding_stock = False
                    if holding_stock == True:
                        buy_price = train_data['adjClose'].iloc[-1]
                        portfolio_returns.append(sell_price / buy_price - 1)

                genome.fitness = np.mean(portfolio_returns)
    
        p_long = Population(config)
        p_short = Population(config_short)
        stats = StatisticsReporter()
        p_long.add_reporter(StdOutReporter(False))
        p_long.add_reporter(stats)
        p_short.add_reporter(StdOutReporter(False))
        p_short.add_reporter(stats)
        winner_long = p_long.run(eval_genomes, 10)
        winner_long_net = FeedForwardNetwork.create(winner_long, config)

        winner_short = p_short.run(eval_genomes_short, 10)
        winner_short_net = FeedForwardNetwork.create(winner_short, config)


        for j in range(6, testing_windows_size):

            optimal_thresholds_long = [int(x*100) for x in winner_long_net.activate((test_data['daily_return_minmax'].iloc[j-5:j].values*100))]
            optimal_thresholds_short = [int(x*100) for x in winner_short_net.activate((test_data['daily_return_minmax'].iloc[j-5:j].values*100))]

            buy_signal_long = (train_data['rsi'].iloc[j] < optimal_thresholds_long[0]) * 1
            sell_signal_long = (train_data['rsi'].iloc[j] > optimal_thresholds_long[1]) * -1

            buy_signal_short = (train_data['rsi'].iloc[j] < optimal_thresholds_long[0]) * 1
            sell_signal_short = (train_data['rsi'].iloc[j] > optimal_thresholds_long[1]) * -1

            if (test_data['adjClose'].iloc[j] > test_data['ema'].iloc[j]):
                if buy_signal_long == 1 and not holding_long:
                    buy_price = test_data.iloc[j]['adjClose']
                    holding_long = True
                elif sell_signal_long == -1 and holding_long:
                    sell_price = test_data.iloc[j]['adjClose']

                    long_capital *= sell_price / buy_price

                    trade_records.append(['Long', buy_price, sell_price, sell_price / buy_price, long_capital])

                    if (sell_price / buy_price) > 1:
                        long_win_num += 1
                    num_long += 1
                    print('Long', buy_price, sell_price, sell_price / buy_price, long_capital)
                    holding_long = False

            elif(test_data['adjClose'].iloc[j] < test_data['ema'].iloc[j]):
                if sell_signal_short == 1 and not holding_short:
                    sell_price = test_data.iloc[j]['adjClose']
                    holding_short = True
                elif buy_signal_short == -1 and holding_short:
                    buy_price = test_data.iloc[j]['adjClose']

                    short_capital *= sell_price / buy_price

                    trade_records.append(['Short', sell_price, buy_price, sell_price / buy_price, short_capital])

                    if (sell_price / buy_price)> 1:
                        short_win_num += 1
                    num_short += 1
                    print('Short', sell_price, buy_price, sell_price / buy_price, short_capital)
                    holding_short = False

        print(long_capital, short_capital)

    columns = ['Type', 'Entry Price', 'Exit Price', 'Profit Factor', 'Capital']
    trade_df = pd.DataFrame(trade_records, columns=columns)
    trade_df.to_csv(f'trade_records_{stock}.csv', index=False)
    long_return = long_capital/10000
    short_return = short_capital/10000
    total_return = (((long_capital+short_capital)/2000)-1)

    if num_long == 0:
        long_win_rate = 0
    else:
        long_win_rate = long_win_num*100/num_long
    
    if num_short == 0:
        short_win_rate =0
    else:
        short_win_rate = short_win_num*100/num_short
    
    result_all.append([stock, long_return*100, num_long, long_win_rate, long_capital, short_return*100, num_short, short_win_rate, short_capital, total_return*100])

columns_all = ['Stock', 'long return(%)', 'Num of Long', 'Long win rate(%)', 'Final Capital of Long', 'short return(%)', 'Num of Short', 'Short win rate(%)', 'Final Capital of Short', 'total return(%)']
result_df = pd.DataFrame(result_all, columns=columns_all)
result_df.to_csv(f'stock_performance.csv', index=False)