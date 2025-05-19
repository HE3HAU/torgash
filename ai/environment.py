import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import sys

logger = logging.getLogger(__name__)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    if not delta.empty: delta.iloc[0] = 0
    gain = delta.where(delta > 0, 0.0).rename("gain")
    loss = (-delta.where(delta < 0, 0.0)).rename("loss")
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.fillna(50.0, inplace=True)
    return rsi

class TradingEnvironment:
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self,
                 historical_data_df: pd.DataFrame,
                 trading_pair_symbol: str,
                 initial_balance_quote: float = 1000.0,
                 trade_fee_percent: float = 0.1, # Остается для расчета чистой стоимости сделки
                 fixed_transaction_penalty: float = 0.0, # <--- НОВЫЙ ПАРАМЕТР: абсолютный штраф за сделку
                 max_steps_per_episode: int = None,
                 window_size: int = 20,
                 sma_short_period: int = 7,
                 sma_long_period: int = 21,
                 rsi_period: int = 14
                 ):
        super(TradingEnvironment, self).__init__()

        self.df = historical_data_df.copy()
        if not isinstance(self.df.index, pd.DatetimeIndex):
            try: self.df.index = pd.to_datetime(self.df.index)
            except Exception as e: logger.error(f"Ошибка преобразования индекса в DatetimeIndex: {e}"); raise

        self.trading_pair_symbol = trading_pair_symbol
        try:
            self.base_currency, self.quote_currency = trading_pair_symbol.split('/')
        except ValueError:
            msg = f"Неверный формат trading_pair_symbol: '{trading_pair_symbol}'. Ожидался 'БАЗА/КОТИРОВКА'."
            logger.error(msg); raise ValueError(msg)

        self.initial_balance_quote = initial_balance_quote
        self.trade_fee_multiplier = 1.0 - (trade_fee_percent / 100.0)
        self.fixed_transaction_penalty = fixed_transaction_penalty # <--- СОХРАНЯЕМ
        self.window_size = window_size

        required_initial_length = max(self.window_size, sma_long_period, rsi_period, 1) + 1
        if self.df.empty or len(self.df) < required_initial_length:
            raise ValueError(f"Данные ({len(self.df)} строк) недостаточны для window_size={self.window_size} и индикаторов (требуется мин. {required_initial_length}).")

        self.df['sma_short'] = self.df['close'].rolling(window=sma_short_period, min_periods=1).mean()
        self.df['sma_long'] = self.df['close'].rolling(window=sma_long_period, min_periods=1).mean()
        self.df['rsi'] = calculate_rsi(self.df['close'], period=rsi_period)
        self.df.dropna(inplace=True)
        
        if len(self.df) < self.window_size:
            raise ValueError(f"После индикаторов и dropna, данных ({len(self.df)}) < window_size ({self.window_size}). Увеличьте объем исходных данных.")

        logger.info(f"Индикаторы (SMA{sma_short_period}, SMA{sma_long_period}, RSI{rsi_period}) рассчитаны. "
                    f"Размер DataFrame после обработки: {len(self.df)}")

        self.action_space_n = 3
        self.observation_space_shape = (window_size + 3 + 2,)

        self._effective_data_len = len(self.df) - self.window_size + 1
        if self._effective_data_len <= 0:
             raise ValueError("Длина данных недостаточна для окна наблюдения после всех обработок.")

        self.max_steps_per_episode = min(self._effective_data_len, max_steps_per_episode) if max_steps_per_episode is not None else self._effective_data_len
        
        self.current_step_in_episode = 0; self.current_df_index = 0
        self.balance_quote = 0.0; self.balance_base = 0.0
        self.net_worth = 0.0; self.last_trade_price = 0.0
        self._current_episode_trades = 0

        logger.info(f"TradingEnvironment: {self.trading_pair_symbol}. Эффективных данных: {self._effective_data_len}. "
                    f"Obs_shape: {self.observation_space_shape}. Max шагов/эпизод: {self.max_steps_per_episode}. "
                    f"Фиксированный штраф за транзакцию: {self.fixed_transaction_penalty:.2f} {self.quote_currency}.")


    def _get_current_price(self) -> float:
        idx_for_price = self.current_df_index + self.window_size - 1
        if 0 <= idx_for_price < len(self.df): return self.df['close'].iloc[idx_for_price]
        logger.error(f"Критическая ошибка индекса цены: idx={idx_for_price}, len_df={len(self.df)}, current_df_idx={self.current_df_index}, window_size={self.window_size}")
        return np.nan

    def _get_observation(self) -> np.ndarray:
        # (Код без изменений, как в последней вашей рабочей версии)
        start_idx = self.current_df_index; end_idx = self.current_df_index + self.window_size - 1
        if not (0 <= start_idx <= end_idx < len(self.df)):
            logger.critical(f"Некорректные индексы для среза наблюдения: start={start_idx}, end={end_idx}, len_df={len(self.df)}")
            return np.zeros(self.observation_space_shape, dtype=np.float32)
        close_prices_window = self.df['close'].iloc[start_idx : end_idx + 1].values
        current_price_at_end = close_prices_window[-1]
        if current_price_at_end > 1e-9: normalized_close_prices = close_prices_window / current_price_at_end
        else: normalized_close_prices = np.zeros_like(close_prices_window)
        sma_short_at_end = self.df['sma_short'].iloc[end_idx]; sma_long_at_end = self.df['sma_long'].iloc[end_idx]
        rsi_at_end = self.df['rsi'].iloc[end_idx]
        norm_sma_short = (sma_short_at_end / current_price_at_end) - 1.0 if current_price_at_end > 1e-9 else 0.0
        norm_sma_long = (sma_long_at_end / current_price_at_end) - 1.0 if current_price_at_end > 1e-9 else 0.0
        norm_rsi = (rsi_at_end - 50.0) / 50.0
        base_value_in_quote = self.balance_base * current_price_at_end if current_price_at_end > 1e-9 else 0.0
        norm_balance_base = base_value_in_quote / self.initial_balance_quote if self.initial_balance_quote > 1e-9 else 0.0
        norm_balance_quote = self.balance_quote / self.initial_balance_quote if self.initial_balance_quote > 1e-9 else 0.0
        observation = np.concatenate((normalized_close_prices, [norm_sma_short], [norm_sma_long], [norm_rsi],
                                      [norm_balance_base], [norm_balance_quote]), axis=0)
        if observation.shape != self.observation_space_shape:
            logger.error(f"Финальная ошибка формы! Ожидалось {self.observation_space_shape}, получено {observation.shape}")
            return np.zeros(self.observation_space_shape, dtype=np.float32)
        return observation.astype(np.float32)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        # (Код без изменений)
        if seed is not None: np.random.seed(seed)
        self.current_step_in_episode = 0; self.current_df_index = 0
        self.balance_quote = self.initial_balance_quote; self.balance_base = 0.0
        self.net_worth = self.initial_balance_quote
        self.last_trade_price = self.df['close'].iloc[self.window_size - 1] if len(self.df) >= self.window_size else 0.0
        self._current_episode_trades = 0
        logger.debug(f"Среда сброшена. Баланс: {self.balance_quote:.2f} {self.quote_currency}, {self.balance_base:.8f} {self.base_currency}")
        observation = self._get_observation(); info = self._get_info()
        return observation, info

    def _take_action(self, action: int) -> float:
        current_price = self._get_current_price()
        transaction_executed = False

        if np.isnan(current_price) or current_price <= 1e-9:
            logger.warning(f"Невалидная цена ({current_price}) на шаге {self.current_step_in_episode}. Действие не выполняется, награда 0.")
            return 0.0

        action_type = action
        prev_net_worth = self.net_worth
        
        if action_type == 1: # Buy
            if self.balance_quote > 1e-8:
                amount_to_buy_base_gross = self.balance_quote / current_price
                amount_to_buy_base_net = amount_to_buy_base_gross * self.trade_fee_multiplier
                if amount_to_buy_base_net > 1e-9:
                    self.balance_base += amount_to_buy_base_net; self.balance_quote = 0.0
                    self.last_trade_price = current_price; self._current_episode_trades += 1; transaction_executed = True
                    logger.debug(f"Эп.шаг {self.current_step_in_episode}: BUY {amount_to_buy_base_net:.8f} {self.base_currency} @ {current_price:.4f}")
                else: logger.debug(f"Эп.шаг {self.current_step_in_episode}: Попытка BUY, объем < min после комиссии.")
            else: logger.debug(f"Эп.шаг {self.current_step_in_episode}: Попытка BUY, нет {self.quote_currency}.")
        elif action_type == 2: # Sell
            if self.balance_base > 1e-9:
                amount_to_sell_base = self.balance_base
                revenue_quote_net = (amount_to_sell_base * current_price) * self.trade_fee_multiplier
                self.balance_quote += revenue_quote_net; sold_base_amount = self.balance_base; self.balance_base = 0.0
                self.last_trade_price = current_price; self._current_episode_trades += 1; transaction_executed = True
                logger.debug(f"Эп.шаг {self.current_step_in_episode}: SELL {sold_base_amount:.8f} {self.base_currency} @ {current_price:.4f}")
            else: logger.debug(f"Эп.шаг {self.current_step_in_episode}: Попытка SELL, нет {self.base_currency}.")
        
        self.net_worth = self.balance_quote + (self.balance_base * current_price)
        reward_from_pnl = self.net_worth - prev_net_worth
        
        penalty = 0.0
        if transaction_executed:
            penalty = self.fixed_transaction_penalty # <--- ИСПОЛЬЗУЕМ ФИКСИРОВАННЫЙ ШТРАФ
            if penalty > 0 : # Логируем, только если штраф есть
                 logger.debug(f"  Применен фиксированный штраф за транзакцию: {penalty:.4f}")

        final_reward = reward_from_pnl - penalty
        return final_reward

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # (Код без изменений, как в последней вашей рабочей версии)
        reward = self._take_action(action); self.current_step_in_episode += 1
        can_get_next_observation = (self.current_df_index + 1 + self.window_size <= len(self.df))
        if can_get_next_observation: self.current_df_index += 1; truncated_by_data_end = False
        else: self.current_df_index = len(self.df) - self.window_size; truncated_by_data_end = True
        if self.current_df_index < 0 : self.current_df_index = 0
        terminated = self.net_worth <= (self.initial_balance_quote * 0.2)
        truncated_by_max_steps = self.current_step_in_episode >= self.max_steps_per_episode
        truncated = truncated_by_max_steps or truncated_by_data_end
        observation = self._get_observation(); info = self._get_info()
        log_msg_suffix = f" Net worth: {self.net_worth:.2f}"
        if terminated: logger.info(f"Эпизод ЗАВЕРШЕН (terminated) на шаге {self.current_step_in_episode}." + log_msg_suffix)
        if truncated and not terminated: logger.info(f"Эпизод ЗАВЕРШЕН (truncated) на шаге {self.current_step_in_episode} (max_steps: {truncated_by_max_steps}, data_end: {truncated_by_data_end})." + log_msg_suffix)
        return observation, reward, terminated, truncated, info

    def _get_info(self) -> dict:
        # (Код без изменений)
        current_price_for_info = self._get_current_price()
        return {'current_step_in_episode': self.current_step_in_episode, 'current_df_index': self.current_df_index,
                'balance_quote': self.balance_quote, 'balance_base': self.balance_base, 'net_worth': self.net_worth,
                'trades_in_episode': self._current_episode_trades,
                'current_price': current_price_for_info if not np.isnan(current_price_for_info) else "N/A"}

    def render(self, mode='human'):
        # (Код без изменений)
        if mode == 'human':
            info = self._get_info()
            price_str = f"{info['current_price']:.4f}" if isinstance(info['current_price'], (int,float)) else info['current_price']
            print(f"Эп.шаг: {info['current_step_in_episode']}, Цена: {price_str}, NetWorth: {info['net_worth']:.2f} {self.quote_currency}, "
                  f"Quote: {info['balance_quote']:.2f}, Base: {info['balance_base']:.8f}, Trades: {info['trades_in_episode']}")

    def close(self): logger.info("TradingEnvironment закрывается.")

# Тестовый блок if __name__ == '__main__':
if __name__ == '__main__':
    test_env_logger = logging.getLogger("TestTradingEnvironmentModule")
    if not test_env_logger.hasHandlers():
        test_env_logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        test_env_logger.addHandler(ch)
    logging.getLogger(__name__).addHandler(ch)
    logging.getLogger(__name__).setLevel(logging.DEBUG)

    data_test_main = {
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='h')),
        'open': np.random.rand(100) * 100 + 1000, 'high': np.random.rand(100) * 120 + 1020,
        'low': np.random.rand(100) * 80 + 980, 'close': np.arange(1000, 1100, 1).astype(float),
        'volume': np.random.rand(100) * 10
    }
    df_for_env_test_main = pd.DataFrame(data_test_main).set_index('timestamp')
    test_env_logger.info("--- Запуск локального теста TradingEnvironment ---")
    try:
        env_instance = TradingEnvironment(
            historical_data_df=df_for_env_test_main, trading_pair_symbol="TEST/USD",
            window_size=5, initial_balance_quote=100.0, trade_fee_percent=0.1,
            fixed_transaction_penalty=1.0, # <--- ТЕСТИРУЕМ С ФИКСИРОВАННЫМ ШТРАФОМ
            sma_short_period=3, sma_long_period=4, rsi_period=4, max_steps_per_episode=20
        )
        test_env_logger.info("Тестовый экземпляр TradingEnvironment успешно создан.")
        obs, info = env_instance.reset()
        # ... (остальной тестовый код, если нужен)
    except Exception as e_gen_test:
        test_env_logger.error(f"Непредвиденная ошибка при тестировании TradingEnvironment: {e_gen_test}", exc_info=True)