                    for j in range(training_window_size):

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