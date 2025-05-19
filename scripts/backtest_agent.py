import logging
import pandas as pd
import os
import sys
import numpy as np # Для np.nan

# Добавляем корневую директорию проекта в PYTHONPATH,
# чтобы можно было импортировать модули из 'core' и 'ai'
# Это необходимо, если вы запускаете скрипт напрямую из директории scripts/
# и структура проекта такая, как мы обсуждали.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # На один уровень выше (из scripts/ в корень проекта)
if project_root not in sys.path:
    sys.path.append(project_root)

from core.utils import setup_logging, safe_float_convert
from ai.agent import RuleBasedAgent
# from ai.agent import RLAgent # Раскомментировать для RL агента

# Настройка логгера для этого скрипта
# Мы не используем setup_logging из utils здесь напрямую, чтобы не зависеть от config.ini для простого скрипта.
# Вместо этого создаем локальный логгер.
script_logger = logging.getLogger('backtest_script')
if not script_logger.hasHandlers(): # Предотвращаем дублирование обработчиков
    script_logger.setLevel(logging.INFO)
    _ch = logging.StreamHandler(sys.stdout)
    _cf = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _ch.setFormatter(_cf)
    script_logger.addHandler(_ch)


def load_historical_data(filepath: str) -> pd.DataFrame | None:
    """
    Загружает исторические данные из CSV файла.
    Ожидается, что первая колонка - это индекс datetime, а также есть колонка 'close'.
    """
    if not os.path.exists(filepath):
        script_logger.error(f"Файл исторических данных не найден: {filepath}")
        return None
    try:
        # Предполагаем, что первая колонка (индекс) - это дата/время
        # и она должна быть правильно распарсена.
        # Если 'timestamp' это имя колонки, а не индекс:
        # df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if 'close' not in df.columns:
            script_logger.error(f"В файле {filepath} отсутствует обязательная колонка 'close'.")
            return None
        # Убедимся, что 'close' содержит числовые данные
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['close'], inplace=True) # Удаляем строки, где 'close' не удалось преобразовать
        script_logger.info(f"Исторические данные загружены из {filepath}, строк: {len(df)}")
        return df
    except Exception as e:
        script_logger.error(f"Ошибка при загрузке или обработке данных из {filepath}: {e}")
        return None


def run_backtest(agent,
                 data_df: pd.DataFrame,
                 initial_balance_quote: float = 1000.0,
                 order_amount_base_currency: float = 0.0, # Если 0, то покупаем/продаем на весь баланс
                 trade_fee_percent: float = 0.1, # Комиссия в процентах, например, 0.1%
                 slippage_percent: float = 0.0 # Проскальзывание в процентах (пока не используется)
                 ) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    """
    Запускает бэктест для заданного агента на исторических данных.

    :param agent: Экземпляр торгового агента (должен иметь метод get_decision(current_price)).
    :param data_df: DataFrame с историческими данными (должен содержать колонку 'close').
    :param initial_balance_quote: Начальный баланс в котируемой валюте.
    :param order_amount_base_currency: Фиксированный размер ордера в базовой валюте.
                                       Если 0, то покупка/продажа на весь доступный баланс.
    :param trade_fee_percent: Комиссия за сделку в процентах.
    :param slippage_percent: Проскальзывание цены (пока не реализовано).
    :return: Кортеж: (DataFrame со сделками, Словарь с итоговой статистикой) или (None, None) в случае ошибки.
    """
    if data_df is None or data_df.empty:
        script_logger.error("Данные для бэктеста отсутствуют или пусты.")
        return None, None

    script_logger.info("=" * 50)
    script_logger.info("Запуск бэктеста...")
    script_logger.info(f"Начальный баланс (котируемая): {initial_balance_quote:.2f}")
    script_logger.info(f"Размер ордера (базовая): {'Весь доступный баланс' if order_amount_base_currency == 0 else f'{order_amount_base_currency:.8f}'}")
    script_logger.info(f"Комиссия за сделку: {trade_fee_percent:.3f}%")
    script_logger.info("=" * 50)


    balance_quote = initial_balance_quote
    balance_base = 0.0
    portfolio_history = [] # История стоимости портфеля
    trades_log = [] # Список для журналирования сделок

    trade_fee_multiplier = 1.0 - (trade_fee_percent / 100.0)

    for timestamp, row in data_df.iterrows():
        current_price = safe_float_convert(row['close'])
        if current_price is None or current_price <=0:
            script_logger.warning(f"{timestamp}: Некорректная цена {row['close']}, пропуск шага.")
            # Обновляем стоимость портфеля на основе последней известной цены, если есть базовая валюта
            portfolio_value = balance_quote + (balance_base * (trades_log[-1]['price'] if trades_log and balance_base > 0 else 0))
            portfolio_history.append({'timestamp': timestamp, 'portfolio_value': portfolio_value})
            continue

        # --- Принятие решения агентом ---
        # Для RL-агента здесь нужно будет формировать state
        # state = create_state_from_row(row, window_data_df, ...)
        # decision = agent.get_decision(state)
        decision = agent.get_decision(current_price) # Для RuleBasedAgent

        # --- Исполнение сделки ---
        executed_price = current_price # В будущем можно добавить модель проскальзывания
        
        if decision == 'buy':
            if balance_quote > 0:
                amount_to_buy_base = 0
                if order_amount_base_currency > 0: # Фиксированный размер
                    # Проверяем, хватает ли средств на фиксированный размер
                    if balance_quote >= (order_amount_base_currency * executed_price):
                        amount_to_buy_base = order_amount_base_currency
                    else:
                        script_logger.debug(f"{timestamp}: Недостаточно средств ({balance_quote:.2f}) для покупки {order_amount_base_currency} BASE по цене {executed_price:.2f}")
                else: # Покупаем на весь доступный баланс котируемой
                    amount_to_buy_base = balance_quote / executed_price
                
                if amount_to_buy_base > 0:
                    actual_bought_base = amount_to_buy_base * trade_fee_multiplier
                    cost_quote = amount_to_buy_base * executed_price # Расход без комиссии

                    balance_base += actual_bought_base
                    balance_quote -= cost_quote # Списываем полную стоимость до комиссии
                    
                    trades_log.append({
                        'timestamp': timestamp, 'type': 'buy', 'price': executed_price,
                        'amount_base': actual_bought_base, 'cost_quote': cost_quote,
                        'fee_percent': trade_fee_percent,
                        'balance_base_after': balance_base, 'balance_quote_after': balance_quote
                    })
                    script_logger.info(f"{timestamp}: BUY {actual_bought_base:.8f} BASE по {executed_price:.2f} (Стоимость: {cost_quote:.2f} QUOTE)")
            else:
                script_logger.debug(f"{timestamp}: Агент решил BUY, но нет QUOTE на балансе.")

        elif decision == 'sell':
            if balance_base > 0:
                amount_to_sell_base = 0
                if order_amount_base_currency > 0: # Фиксированный размер
                    if balance_base >= order_amount_base_currency:
                        amount_to_sell_base = order_amount_base_currency
                    else: # Если не хватает на фикс. размер, продаем все что есть (альтернатива - не продавать)
                        amount_to_sell_base = balance_base
                        script_logger.debug(f"{timestamp}: Недостаточно BASE ({balance_base:.8f}) для продажи фиксированного объема {order_amount_base_currency}. Продаем все.")
                else: # Продаем весь доступный баланс базовой
                    amount_to_sell_base = balance_base
                
                if amount_to_sell_base > 0:
                    revenue_quote = (amount_to_sell_base * executed_price) * trade_fee_multiplier
                    
                    balance_base -= amount_to_sell_base # Списываем проданное
                    balance_quote += revenue_quote
                    
                    trades_log.append({
                        'timestamp': timestamp, 'type': 'sell', 'price': executed_price,
                        'amount_base': amount_to_sell_base, 'revenue_quote': revenue_quote,
                        'fee_percent': trade_fee_percent,
                        'balance_base_after': balance_base, 'balance_quote_after': balance_quote
                    })
                    script_logger.info(f"{timestamp}: SELL {amount_to_sell_base:.8f} BASE по {executed_price:.2f} (Доход: {revenue_quote:.2f} QUOTE)")
            else:
                script_logger.debug(f"{timestamp}: Агент решил SELL, но нет BASE на балансе.")
        
        # Запись текущей стоимости портфеля
        current_portfolio_value = balance_quote + (balance_base * current_price)
        portfolio_history.append({'timestamp': timestamp, 'portfolio_value': current_portfolio_value})

    script_logger.info("=" * 50)
    script_logger.info("Бэктест завершен.")

    # --- Расчет итоговой статистики ---
    final_portfolio_value = portfolio_history[-1]['portfolio_value'] if portfolio_history else initial_balance_quote
    total_return_percent = ((final_portfolio_value / initial_balance_quote) - 1) * 100 if initial_balance_quote > 0 else 0
    
    num_trades = len(trades_log)
    # Дополнительные метрики можно добавить здесь (Sharpe ratio, Max Drawdown, etc.)
    # Для Max Drawdown:
    portfolio_df = pd.DataFrame(portfolio_history).set_index('timestamp')
    peak = portfolio_df['portfolio_value'].expanding(min_periods=1).max()
    drawdown = (portfolio_df['portfolio_value'] - peak) / peak
    max_drawdown_percent = drawdown.min() * 100 if not drawdown.empty else 0


    stats = {
        'start_date': data_df.index[0].strftime('%Y-%m-%d %H:%M:%S') if not data_df.empty else "N/A",
        'end_date': data_df.index[-1].strftime('%Y-%m-%d %H:%M:%S') if not data_df.empty else "N/A",
        'duration_days': (data_df.index[-1] - data_df.index[0]).days if not data_df.empty and len(data_df)>1 else 0,
        'initial_portfolio_value': initial_balance_quote,
        'final_portfolio_value': final_portfolio_value,
        'total_return_usd': final_portfolio_value - initial_balance_quote,
        'total_return_percent': total_return_percent,
        'number_of_trades': num_trades,
        'max_drawdown_percent': max_drawdown_percent
    }

    script_logger.info("Итоговая статистика:")
    for key, value in stats.items():
        if isinstance(value, float):
            script_logger.info(f"  {key.replace('_', ' ').capitalize()}: {value:.2f}")
        else:
            script_logger.info(f"  {key.replace('_', ' ').capitalize()}: {value}")
    script_logger.info("=" * 50)

    return pd.DataFrame(trades_log), stats


if __name__ == "__main__":
    script_logger.info("Запуск скрипта бэктестинга...")

    # --- Параметры бэктеста ---
    # Путь к файлу с историческими данными (замените на свой)
    # Предполагается, что файл лежит в data/historical/
    data_filename = "binance_BTCUSDT_1h.csv" # Пример файла, который может создать download_data.py
    historical_data_filepath = os.path.join(project_root, "data", "historical", data_filename)

    initial_balance = 10000.0  # Начальный баланс в USDT
    # Пороги для RuleBasedAgent (можно загружать из конфига или задавать здесь)
    buy_thresh = 60000.0
    sell_thresh = 65000.0
    # Размер ордера (0 = на весь баланс, >0 = фиксированный размер в базовой валюте)
    order_amount = 0 #0.001
    fee = 0.075 # Комиссия Binance Spot (тейкер)

    # 1. Загрузка данных
    data = load_historical_data(historical_data_filepath)

    if data is not None:
        # 2. Инициализация агента
        # Передаем логгер скрипта агенту, чтобы его сообщения тоже выводились
        agent = RuleBasedAgent(buy_threshold=buy_thresh,
                               sell_threshold=sell_thresh,
                               logger_instance=script_logger)
        script_logger.info(f"Агент RuleBasedAgent инициализирован с порогами: покупка < {buy_thresh}, продажа > {sell_thresh}")

        # 3. Запуск бэктеста
        trades, statistics = run_backtest(agent,
                                          data,
                                          initial_balance_quote=initial_balance,
                                          order_amount_base_currency=order_amount,
                                          trade_fee_percent=fee)

        if trades is not None:
            script_logger.info(f"Количество совершенных сделок: {len(trades)}")
            # Можно сохранить лог сделок в CSV, если нужно
            # trades_csv_path = os.path.join(project_root, "data", "backtest_trades_log.csv")
            # trades.to_csv(trades_csv_path)
            # script_logger.info(f"Лог сделок сохранен в: {trades_csv_path}")

            # Здесь можно добавить код для построения графиков (например, с matplotlib)
            # - график стоимости портфеля
            # - график цены с отметками сделок
            pass
    else:
        script_logger.error("Не удалось загрузить исторические данные. Бэктест не может быть запущен.")

    script_logger.info("Скрипт бэктестинга завершил работу.")