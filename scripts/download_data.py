import ccxt
import pandas as pd
import time
import os
import logging
import sys
import traceback
from datetime import datetime, timedelta, timezone

# Добавляем корневую директорию проекта в PYTHONPATH (если необходимо для импортов из других модулей)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# if project_root not in sys.path:
#     sys.path.append(project_root)
# from core.utils import ... # Если нужны какие-то утилиты оттуда

# Настройка логгера для этого скрипта
script_logger = logging.getLogger('download_data_script')
if not script_logger.hasHandlers(): # Предотвращаем дублирование обработчиков
    script_logger.setLevel(logging.INFO)
    _ch = logging.StreamHandler(sys.stdout)
    _cf = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _ch.setFormatter(_cf)
    script_logger.addHandler(_ch)


def fetch_ohlcv_data(exchange_id: str,
                     symbol: str,
                     timeframe: str = '1h',
                     since_datetime_str: str = None, # Дата начала в формате 'YYYY-MM-DD HH:MM:SS' или 'YYYY-MM-DD'
                     limit_per_fetch: int = 500,   # Количество свечей за один запрос
                     max_retries: int = 3,
                     delay_between_retries_sec: int = 10
                     ) -> pd.DataFrame | None:
    """
    Загружает OHLCV данные с указанной биржи для заданного символа и таймфрейма.

    :param exchange_id: ID биржи (например, 'binance', 'bybit', 'okx').
    :param symbol: Торговая пара (например, 'BTC/USDT').
    :param timeframe: Таймфрейм (например, '1m', '5m', '1h', '1d').
    :param since_datetime_str: Дата и время начала загрузки данных в формате 'YYYY-MM-DD HH:MM:SS' или 'YYYY-MM-DD'.
                               Если None, загружает данные за определенный период в прошлом (например, последний год).
    :param limit_per_fetch: Максимальное количество свечей, запрашиваемых за один вызов API.
    :param max_retries: Максимальное количество попыток при сетевых ошибках.
    :param delay_between_retries_sec: Задержка между повторными попытками в секундах.
    :return: DataFrame pandas с данными OHLCV или None в случае ошибки.
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
    except AttributeError:
        script_logger.error(f"Биржа '{exchange_id}' не найдена в ccxt.")
        return None

    exchange = exchange_class() # Для публичных данных API ключи обычно не нужны

    if not exchange.has.get('fetchOHLCV'):
        script_logger.error(f"Биржа {exchange_id} не поддерживает метод fetchOHLCV.")
        return None

    # Преобразование строки даты в timestamp (ms)
    since_timestamp_ms = None
    if since_datetime_str:
        try:
            # Пытаемся распарсить с временем, потом без времени
            dt_obj = datetime.strptime(since_datetime_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                dt_obj = datetime.strptime(since_datetime_str, '%Y-%m-%d')
            except ValueError:
                script_logger.error(f"Неверный формат даты начала '{since_datetime_str}'. "
                                   "Используйте 'YYYY-MM-DD' или 'YYYY-MM-DD HH:MM:SS'.")
                return None
        # Убедимся, что datetime объект осведомлен о временной зоне (предполагаем UTC)
        since_timestamp_ms = int(dt_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
    else:
        # Если дата не указана, загружаем, например, за последние 365 дней
        default_start_dt = datetime.now(timezone.utc) - timedelta(days=365)
        since_timestamp_ms = int(default_start_dt.timestamp() * 1000)
        script_logger.info(f"Дата начала не указана. Загрузка данных за последние ~365 дней, начиная с {default_start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")


    all_ohlcv = []
    current_fetch_timestamp_ms = since_timestamp_ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    script_logger.info(f"Начало загрузки данных для {symbol} с {exchange_id} (таймфрейм: {timeframe})")
    script_logger.info(f"Период загрузки: с {datetime.fromtimestamp(current_fetch_timestamp_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} "
                       f"по {datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    retries = 0
    while current_fetch_timestamp_ms < now_ms:
        try:
            script_logger.debug(f"Запрос {limit_per_fetch} свечей, начиная с {datetime.fromtimestamp(current_fetch_timestamp_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            # Некоторые биржи могут иметь ограничение на количество возвращаемых свечей,
            # даже если limit_per_fetch больше. ccxt обычно это учитывает.
            ohlcv_batch = exchange.fetch_ohlcv(symbol, timeframe, current_fetch_timestamp_ms, limit_per_fetch)
            retries = 0 # Сброс счетчика попыток при успешном запросе

            if not ohlcv_batch: # Если биржа вернула пустой список
                script_logger.info("Биржа вернула пустой список свечей. Возможно, достигнут конец доступных данных.")
                break

            # Убираем дубликаты, если последняя свеча предыдущего батча совпадает с первой текущего
            if all_ohlcv and ohlcv_batch and all_ohlcv[-1][0] == ohlcv_batch[0][0]:
                ohlcv_batch = ohlcv_batch[1:] # Удаляем первую свечу текущего батча
                if not ohlcv_batch: continue # Если после удаления батч пуст

            all_ohlcv.extend(ohlcv_batch)
            last_timestamp_in_batch_ms = ohlcv_batch[-1][0]
            
            script_logger.info(f"Загружено {len(ohlcv_batch)} свечей. "
                               f"Последняя метка времени в батче: {datetime.fromtimestamp(last_timestamp_in_batch_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}. "
                               f"Всего загружено: {len(all_ohlcv)} свечей.")

            if last_timestamp_in_batch_ms >= now_ms:
                script_logger.info("Достигнуто текущее время. Загрузка завершена.")
                break
            
            # Переходим к следующей порции данных
            # timeframe_duration_ms = exchange.parse_timeframe(timeframe) * 1000
            # current_fetch_timestamp_ms = last_timestamp_in_batch_ms + timeframe_duration_ms # Можно так, но безопаснее от последней свечи
            current_fetch_timestamp_ms = last_timestamp_in_batch_ms + 1 # +1 мс, чтобы избежать дублирования первой свечи следующего батча

            # Уважение к лимитам API биржи
            if hasattr(exchange, 'rateLimit'):
                time.sleep(exchange.rateLimit / 1000)

        except ccxt.NetworkError as e:
            retries += 1
            script_logger.warning(f"Сетевая ошибка: {e}. Попытка {retries}/{max_retries} через {delay_between_retries_sec} секунд...")
            if retries >= max_retries:
                script_logger.error("Превышено максимальное количество попыток при сетевой ошибке.")
                return None
            time.sleep(delay_between_retries_sec)
        except ccxt.ExchangeError as e:
            script_logger.error(f"Ошибка биржи: {e}. Прекращение загрузки для {symbol}.")
            # Здесь можно решить, возвращать ли уже загруженные данные или None
            break # Прерываем цикл, но данные в all_ohlcv остаются
        except Exception as e:
            script_logger.error(f"Непредвиденная ошибка: {e}. Прекращение загрузки.")
            script_logger.debug(traceback.format_exc())
            return None # При серьезной ошибке лучше вернуть None

    if not all_ohlcv:
        script_logger.warning(f"Не удалось загрузить данные OHLCV для {symbol} на {exchange_id}.")
        return None

    # Преобразование в DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) # Указываем, что timestamp это UTC
    df.set_index('timestamp', inplace=True)

    # Удаление дубликатов по индексу (на всякий случай, если логика выше не сработала идеально)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True) # Гарантируем сортировку по времени

    script_logger.info(f"Загрузка данных для {symbol} ({timeframe}) завершена. Всего {len(df)} свечей.")
    return df


if __name__ == "__main__":
    script_logger.info("Запуск скрипта загрузки исторических данных...")

    # --- Параметры загрузки ---
    exchange_to_use = 'binance' # Примеры: 'bybit', 'okx', 'kraken', 'coinbasepro'
    symbol_to_download = 'BTC/USDT'
    timeframe_to_download = '1h'    # Примеры: '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M'
    
    # Дата начала загрузки (YYYY-MM-DD или YYYY-MM-DD HH:MM:SS). Если None, то за последний год.
    # start_date_str = "2022-01-01"
    start_date_str = None # Загрузит за последние ~365 дней

    # Директория для сохранения данных
    # ../data/historical/ относительно текущего скрипта
    project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_directory = os.path.join(project_root_dir, "data", "historical")
    os.makedirs(save_directory, exist_ok=True) # Создаем директорию, если ее нет

    # Формирование имени файла
    # Заменяем '/' в символе на '_', чтобы избежать проблем с путем к файлу
    filename_symbol_part = symbol_to_download.replace('/', '_')
    output_filename = f"{exchange_to_use}_{filename_symbol_part}_{timeframe_to_download}.csv"
    output_filepath = os.path.join(save_directory, output_filename)

    # Загрузка данных
    ohlcv_df = fetch_ohlcv_data(exchange_id=exchange_to_use,
                                symbol=symbol_to_download,
                                timeframe=timeframe_to_download,
                                since_datetime_str=start_date_str,
                                limit_per_fetch=1000) # Binance позволяет до 1000, Bybit до 1000 для спота

    if ohlcv_df is not None and not ohlcv_df.empty:
        try:
            ohlcv_df.to_csv(output_filepath)
            script_logger.info(f"Данные успешно сохранены в файл: {output_filepath}")
        except Exception as e:
            script_logger.error(f"Ошибка при сохранении данных в CSV файл {output_filepath}: {e}")
    else:
        script_logger.warning("Не удалось загрузить данные или DataFrame пуст. Файл не будет создан/перезаписан.")

    script_logger.info("Скрипт загрузки данных завершил работу.")