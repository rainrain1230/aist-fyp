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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

random.seed(9001)
training_window_size = 30
testing_windows_size = 10

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
    
    num_long = 0
    num_short = 0
    holding_long = False
    holding_short = False
    buy_price = 0
    long_win_num = 0
    short_win_num = 0

    trade_records = []

    for i in range(0, len(data) - training_window_size-testing_windows_size, training_window_size):
        train_data = data.iloc[i:i+training_window_size].copy()
        test_data = data.iloc[i+training_window_size+1:i+training_window_size+1+testing_windows_size].copy()

        train_data['daily_return_minmax'] = MinMaxScaler().fit_transform(train_data['daily_return'].values.reshape(-1, 1))

        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                net = FeedForwardNetwork.create(genome, config)
                holding_stock = False
                buy_pirce = 0
                total_return = 1

                thresholds = [int(x*100) for x in net.activate(np.array(train_data['adjClose_minmax']))]

                if thresholds[0] > thresholds[1] or 10 > thresholds[0] or 90 < thresholds[1]:
                    genome.fitness = 0
                else:
                    for j in range(training_window_size):
                        train_data.loc[:, 'ema'] = ta.ema(train_data['adjClose'], length=(thresholds[2]//10))
                        rsi = ta.rsi(train_data['adjClose'], length=(thresholds[3]//10))

                        buy_signal = (rsi < thresholds[0]) * 1
                        sell_signal = (rsi > thresholds[1]) * -1


                        if train_data['adjClose'].iloc[j] > train_data['ema'].iloc[j]:
                            if buy_signal.any() and not holding_stock:
                                buy_price = train_data.loc[buy_signal.idxmax(), 'adjClose']
                                holding_stock = True
                            elif sell_signal.any and holding_stock:
                                sell_price = train_data.loc[sell_signal.idxmax(), 'adjClose']
                                total_return *= sell_price / buy_price
                                holding_stock = False

                genome.fitness = total_return

        def eval_genomes_short(genomes, config):
            for genome_id, genome in genomes:
                net = FeedForwardNetwork.create(genome, config)
                holding_stock = False
                buy_pirce = 0
                total_return = 1

                thresholds = [int(x*100) for x in net.activate(np.array(train_data['adjClose_minmax']))]

                if thresholds[0] > thresholds[1] or 10 > thresholds[0] or 90 < thresholds[1]:
                    genome.fitness = 0
                else:
                    for j in range(training_window_size):
                        train_data.loc[:, 'ema'] = ta.ema(train_data['adjClose'], length=(thresholds[2]//10))
                        rsi = ta.rsi(train_data['adjClose'], length=(thresholds[3]//10))

                        buy_signal = (rsi < thresholds[0]) * 1
                        sell_signal = (rsi > thresholds[1]) * -1

                        if train_data['adjClose'].iloc[j] < train_data['ema'].iloc[j]:
                            if sell_signal.any and not holding_stock:
                                sell_price = train_data.loc[sell_signal.idxmax(), 'adjClose']
                                holding_stock = True
                            elif buy_signal.any() and holding_stock:
                                buy_price = train_data.loc[buy_signal.idxmax(), 'adjClose']
                                total_return *= sell_price / buy_price
                                holding_stock = False
                    
                genome.fitness = total_return
    
    
        p_long = Population(config)
        p_short = Population(config)
        stats = StatisticsReporter()
        p_long.add_reporter(StdOutReporter(False))
        p_long.add_reporter(stats)
        p_short.add_reporter(StdOutReporter(False))
        p_short.add_reporter(stats)
        winner_long = p_long.run(eval_genomes, 20)
        winner_long_net = FeedForwardNetwork.create(winner_long, config)

        winner_short = p_short.run(eval_genomes_short, 20)
        winner_short_net = FeedForwardNetwork.create(winner_short, config)

        optimal_thresholds_long = [int(x*100) for x in winner_long_net.activate(test_data['adjClose_minmax'].values)]
        optimal_thresholds_short = [int(x*100) for x in winner_short_net.activate(test_data['adjClose_minmax'].values)]

        test_data.loc[:, 'ema_long'] = ta.ema(test_data['adjClose'], length=(optimal_thresholds_long[2]//10))
        rsi_long = ta.rsi(test_data['adjClose'], length=(optimal_thresholds_long[3]//10))

        test_data.loc[:, 'ema_short'] = ta.ema(test_data['adjClose'], length=(optimal_thresholds_short[2]//10))
        rsi_short = ta.rsi(test_data['adjClose'], length=(optimal_thresholds_short[3]//10))

        for j in range(training_window_size):
            if (test_data['adjClose'].iloc[j] > test_data['ema_long'].iloc[j]):
                if (rsi_long < optimal_thresholds_long[0]).any() and not holding_long:
                    buy_price = test_data.iloc[j]['adjClose']
                    holding_long = True
                elif (rsi_long > optimal_thresholds_long[1]).any() and holding_long:
                    sell_price = test_data.iloc[j]['adjClose']

                    long_capital *= sell_price / buy_price

                    trade_records.append(['Long', buy_price, sell_price, sell_price / buy_price, long_capital])

                    if (sell_price / buy_price) > 1:
                        long_win_num += 1
                    num_long += 1
                    print('Long', buy_price, sell_price, sell_price / buy_price, long_capital)
                    holding_long = False

            elif(test_data['adjClose'].iloc[j] < test_data['ema_short'].iloc[j]):
                if (rsi_short > optimal_thresholds_short[1]).any() and not holding_short:
                    sell_price = test_data.iloc[j]['adjClose']
                    holding_short = True
                elif (rsi_short < optimal_thresholds_short[0]).any() and holding_short:
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
    long_return = long_capital/1000
    short_return = short_capital/1000
    total_return = (((long_capital+short_capital)/2000)-1)

    result_all.append([stock, long_return*100, num_long, long_win_num*100/num_long, long_capital, short_return*100, num_short, short_win_num*100/num_short, short_capital, total_return*100])

columns_all = ['Stock', 'long return(%)', 'Num of Long', 'Long win rate(%)', 'Final Capital of Long', 'short return(%)', 'Num of Short', 'Short win rate(%)', 'Final Capital of Short', 'total return(%)']
result_df = pd.DataFrame(result_all, columns=columns_all)
result_df.to_csv(f'stock_performance_v24.csv', index=False)