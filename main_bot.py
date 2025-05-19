import time
import logging
import traceback
import os
import sys

# Добавляем корневую директорию проекта в PYTHONPATH, чтобы main_bot.py,
# находясь в корне, мог корректно импортировать из подпапок ai, core
# Это может быть излишним, если вы запускаете из корня проекта, и PYTHONPATH уже настроен IDE,
# но для надежности при запуске из командной строки из любой директории - полезно.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Вставляем в начало, чтобы иметь приоритет

# --- Импорт наших модулей ---
from core.config_loader import (
    load_dotenv_file,
    load_ini_config,
    get_api_keys_env_names,
    get_bot_settings,
    get_logging_settings
)
from core.exchange_client import ExchangeClient
from core.utils import setup_logging
from ai.agent import RuleBasedAgent
# from ai.agent import YourRLAgentClass # Для будущего RL агента

# --- Предварительная настройка логгера (до чтения конфига) ---
# Это позволит логировать ошибки на самых ранних этапах, например, если конфиг не найден.
# Мы настроим корневой логгер, так как имя главного логгера еще не известно.
# setup_logging из utils очистит существующие обработчики, так что дублирования не будет.
temp_logger_setup_done = False
try:
    # Пытаемся настроить минимальный логгер сразу
    # Используем logger_name=None для корневого логгера
    pre_logger = setup_logging(log_level_numeric=logging.INFO,
                               log_level_string="INFO",
                               log_to_console=True,
                               logger_name=None, # Настраиваем корневой логгер
                               clear_existing_handlers=True)
    pre_logger.info("[PRE-CONFIG] Предварительный логгер инициализирован.")
    temp_logger_setup_done = True
except Exception as e_pre_log:
    print(f"CRITICAL ERROR during pre-logging setup: {e_pre_log}", file=sys.stderr)
    # Если даже базовый логгер не настроился, дальнейшая работа бессмысленна
    # или будет без логов. Можно здесь выйти, но попробуем продолжить.


# --- ГЛАВНАЯ ФУНКЦИЯ БОТА ---
def run_bot():
    """
    Основная функция, запускающая и управляющая торговым ботом.
    """
    logger = logging.getLogger("MainBot") # Получаем наш основной логгер (он уже настроен или будет настроен)
                                         # Если pre_logger был корневым, этот логгер унаследует его настройки,
                                         # пока мы не перенастроим его по конфигу.

    try:
        # 1. Загрузка конфигурации
        logger.info("Загрузка конфигурации...")
        # Ищем .env и config.ini относительно main_bot.py (т.е. в корне проекта)
        env_loaded = load_dotenv_file(dotenv_path=os.path.join(project_root, '.env'))
        
        config_ini_path = os.path.join(project_root, 'config.ini')
        app_config = load_ini_config(config_ini_path)

        if not app_config:
            logger.critical(f"Не удалось загрузить конфигурационный файл: {config_ini_path}. Выход.")
            return # Выход, если конфиг не загружен

        # 2. Настройка логирования на основе конфигурации
        # setup_logging вызывается еще раз, чтобы применить настройки из config.ini
        # Он очистит предыдущие обработчики корневого логгера и настроит заново.
        log_settings = get_logging_settings(app_config)
        # Настраиваем корневой логгер по параметрам из конфига.
        # Все дочерние логгеры (включая "MainBot") унаследуют его настройки.
        final_logger = setup_logging(log_level_numeric=log_settings['log_level_numeric'],
                                     log_level_string=log_settings['log_level_string'],
                                     log_file=os.path.join(project_root, log_settings['log_file']), # Путь к лог-файлу из корня
                                     log_to_console=True, # Предполагаем, что всегда хотим в консоль
                                     logger_name=None, # Перенастраиваем корневой логгер
                                     clear_existing_handlers=True)
        logger = logging.getLogger("MainBot") # Обновляем ссылку на наш логгер после перенастройки
        logger.info("Логгер успешно перенастроен согласно конфигурационному файлу.")
        if not env_loaded: # Логируем после настройки основного логгера
            logger.warning(".env файл не был загружен. Убедитесь, что он существует или переменные окружения установлены в ОС.")


        # 3. Получение настроек
        bot_settings = get_bot_settings(app_config)
        api_env_names = get_api_keys_env_names(app_config)

        if not bot_settings: # Если get_bot_settings вернул пустой словарь из-за отсутствия секции
            logger.critical("Секция [BOT_SETTINGS] отсутствует или невалидна в config.ini. Выход.")
            return

        # Извлечение API ключей из переменных окружения
        # Пока что ориентируемся на Bybit, как в вашем изначальном коде
        # TODO: Добавить логику для выбора биржи и соответствующих ключей из api_env_names
        current_exchange_name = bot_settings.get('exchange_name', 'bybit')
        api_key = None
        api_secret = None
        api_password = None # Для OKX

        if current_exchange_name == 'bybit':
            api_key = os.getenv(api_env_names['bybit_api_key_env'])
            api_secret = os.getenv(api_env_names['bybit_api_secret_env'])
        elif current_exchange_name == 'okx':
            api_key = os.getenv(api_env_names['okx_api_key_env'])
            api_secret = os.getenv(api_env_names['okx_api_secret_env'])
            api_password = os.getenv(api_env_names['okx_api_password_env'])
        # Добавьте другие биржи по аналогии

        if not api_key or not api_secret:
            key_name_used = api_env_names.get(f"{current_exchange_name}_api_key_env", "N/A")
            secret_name_used = api_env_names.get(f"{current_exchange_name}_api_secret_env", "N/A")
            logger.critical(f"API ключ или секрет для биржи {current_exchange_name} не найдены в переменных окружения. "
                            f"Ожидались переменные: {key_name_used}, {secret_name_used}. Выход.")
            return

        # 4. Инициализация клиента биржи
        logger.info(f"Инициализация клиента для биржи: {current_exchange_name}")
        exchange_client = ExchangeClient(
            exchange_name=current_exchange_name,
            api_key=api_key,
            api_secret=api_secret,
            api_password=api_password, # Будет None, если не для OKX
            is_testnet=True,  # TODO: Сделать настраиваемым через config.ini
            logger_instance=logging.getLogger("ExchangeClient"), # Именованный логгер для клиента
            default_spot_category='spot' # Или из конфига
        )
        if not exchange_client.initialize():
            logger.critical("Не удалось инициализировать клиент биржи. Выход.")
            return
        logger.info("Клиент биржи успешно инициализирован.")

        # 5. Инициализация торгового агента
        # Пока используем RuleBasedAgent, параметры берем из bot_settings
        agent = RuleBasedAgent(
            buy_threshold=bot_settings['buy_price_threshold'],
            sell_threshold=bot_settings['sell_price_threshold'],
            logger_instance=logging.getLogger("TradingAgent")
        )
        logger.info(f"Торговый агент ({type(agent).__name__}) инициализирован.")

        # --- Подготовка к основному циклу ---
        trading_pair = bot_settings['trading_pair']
        base_currency = trading_pair.split('/')[0]
        quote_currency = trading_pair.split('/')[1]
        order_amount = bot_settings['order_amount_base_currency']

        logger.info(f"Бот сконфигурирован для торговли парой {trading_pair} на бирже {current_exchange_name}.")
        logger.info(f"Стратегия: Покупка < {bot_settings['buy_price_threshold']}, Продажа > {bot_settings['sell_price_threshold']}")
        logger.info(f"Размер ордера: {order_amount} {base_currency}")

        # Проверка начальных балансов
        initial_base_balance = exchange_client.get_balance(base_currency)
        initial_quote_balance = exchange_client.get_balance(quote_currency)
        if initial_base_balance is None or initial_quote_balance is None:
            logger.error("Не удалось получить начальные балансы. Проверьте API ключи и их права. Бот не будет запущен.")
            return
        logger.info(f"Начальный баланс: {initial_base_balance:.8f} {base_currency}, {initial_quote_balance:.2f} {quote_currency}")


        active_order_details = None # {'id': '123', 'side': 'buy', 'price': 60000.0, 'amount': 0.001}

        # --- Основной цикл бота ---
        logger.info("--- Запуск основного цикла бота ---")
        while True:
            try:
                current_price = exchange_client.get_current_price(trading_pair)
                if current_price is None:
                    logger.warning(f"Не удалось получить текущую цену для {trading_pair}. Пауза и следующая итерация.")
                    time.sleep(bot_settings['main_loop_pause_seconds'])
                    continue
                logger.info(f"Текущая цена {trading_pair}: {current_price:.2f} {quote_currency}")

                current_order_status_for_logic = None

                # A. Проверка статуса активного ордера (если есть)
                if active_order_details and 'id' in active_order_details:
                    order_id = active_order_details['id']
                    logger.info(f"Проверка статуса активного ордера ID: {order_id} ({active_order_details.get('side')})")
                    status = exchange_client.get_order_status_by_id(order_id, trading_pair)
                    logger.info(f"Статус ордера {order_id} от биржи: {status}")
                    current_order_status_for_logic = status

                    if status == 'closed': # 'closed' у ccxt часто означает исполнен (filled)
                        logger.info(f"Ордер {order_id} ({active_order_details.get('side')}) ИСПОЛНЕН!")
                        # Здесь можно добавить логику после исполнения (уведомления, запись в БД и т.д.)
                        active_order_details = None
                    elif status == 'canceled':
                        logger.info(f"Ордер {order_id} ({active_order_details.get('side')}) ОТМЕНЕН.")
                        active_order_details = None
                    elif status in ['rejected', 'expired', 'not_found_or_too_old', None]: # None если была ошибка получения статуса
                        logger.warning(f"Ордер {order_id} ({active_order_details.get('side')}) не найден/отклонен/истек "
                                       f"или не удалось получить статус ({status}). Сброс отслеживания.")
                        active_order_details = None
                    elif status in ['open', 'new', 'partially_filled']:
                        logger.info(f"Ордер {order_id} ({active_order_details.get('side')}) все еще активен (статус: {status}). Ожидание...")
                        # Можно добавить логику изменения ордера, если цена ушла слишком далеко (order chasing)
                    else: # Неизвестный или непредвиденный статус
                        logger.warning(f"Неизвестный или непредвиденный статус '{status}' для ордера {order_id}. Сброс отслеживания.")
                        active_order_details = None

                # B. Логика принятия решений и создания нового ордера (только если нет активного)
                if not active_order_details:
                    decision = agent.get_decision(current_price) # Агент принимает решение

                    if decision == 'buy':
                        logger.info(f"Агент принял решение: BUY по цене {current_price:.2f}")
                        # Проверка баланса котируемой валюты
                        quote_bal = exchange_client.get_balance(quote_currency)
                        required_quote = order_amount * current_price # Приблизительная стоимость
                        if quote_bal is not None and quote_bal >= required_quote:
                            logger.info(f"Достаточно {quote_currency} ({quote_bal:.2f}) для покупки {order_amount} {base_currency} (требуется ~{required_quote:.2f}).")
                            # Цена для лимитного ордера. Можно использовать current_price или немного лучше (для покупателя).
                            # Для простоты пока используем current_price.
                            order_to_place_price = current_price
                            new_order = exchange_client.create_limit_order(trading_pair, 'buy', order_amount, order_to_place_price)
                            if new_order and 'id' in new_order:
                                active_order_details = new_order # Сохраняем весь объект ордера
                                logger.info(f"Создан новый BUY ордер ID: {active_order_details['id']} по цене {order_to_place_price}")
                            else:
                                logger.error("Не удалось создать BUY ордер или ордер не вернул ID.")
                        elif quote_bal is None:
                             logger.warning(f"Не удалось получить баланс {quote_currency} для совершения покупки.")
                        else:
                            logger.info(f"Недостаточно {quote_currency} ({quote_bal:.2f}) для покупки. Требуется ~{required_quote:.2f}.")

                    elif decision == 'sell':
                        logger.info(f"Агент принял решение: SELL по цене {current_price:.2f}")
                        # Проверка баланса базовой валюты
                        base_bal = exchange_client.get_balance(base_currency)
                        if base_bal is not None and base_bal >= order_amount:
                            logger.info(f"Достаточно {base_currency} ({base_bal:.8f}) для продажи {order_amount} {base_currency}.")
                            order_to_place_price = current_price # Аналогично, для простоты
                            new_order = exchange_client.create_limit_order(trading_pair, 'sell', order_amount, order_to_place_price)
                            if new_order and 'id' in new_order:
                                active_order_details = new_order
                                logger.info(f"Создан новый SELL ордер ID: {active_order_details['id']} по цене {order_to_place_price}")
                            else:
                                logger.error("Не удалось создать SELL ордер или ордер не вернул ID.")
                        elif base_bal is None:
                            logger.warning(f"Не удалось получить баланс {base_currency} для совершения продажи.")
                        else:
                            logger.info(f"Недостаточно {base_currency} ({base_bal:.8f}) для продажи. Требуется: {order_amount}.")
                    else: # decision == 'hold'
                        logger.info("Агент принял решение: HOLD. Новый ордер не требуется.")

                # C. Пауза
                is_active_open_order_tracked = active_order_details and \
                                              current_order_status_for_logic in ['open', 'new', 'partially_filled']
                
                current_pause_seconds = bot_settings['order_check_interval_seconds'] if is_active_open_order_tracked \
                                        else bot_settings['main_loop_pause_seconds']
                
                logger.debug(f"Пауза на {current_pause_seconds} секунд... "
                            f"(Активный ордер отслеживается и открыт: {is_active_open_order_tracked})")
                time.sleep(current_pause_seconds)

            except KeyboardInterrupt:
                logger.info("Получен сигнал KeyboardInterrupt. Завершение работы бота...")
                if active_order_details and active_order_details.get('id'):
                    order_id_on_exit = active_order_details.get('id')
                    logger.info(f"Проверка статуса ордера {order_id_on_exit} перед возможной отменой.")
                    status_before_cancel = exchange_client.get_order_status_by_id(order_id_on_exit, trading_pair)
                    if status_before_cancel in ['open', 'new', 'partially_filled']:
                        logger.info(f"Активный ордер {order_id_on_exit} (статус: {status_before_cancel}) будет отменен.")
                        cancelled_on_exit = exchange_client.cancel_order_by_id(order_id_on_exit, trading_pair)
                        if cancelled_on_exit:
                            logger.info(f"Ордер {order_id_on_exit} успешно отменен при выходе.")
                        else:
                            logger.error(f"Не удалось отменить ордер {order_id_on_exit} при выходе.")
                    else:
                        logger.info(f"Ордер {order_id_on_exit} уже не в активном состоянии ({status_before_cancel}), отмена не требуется.")
                break # Выход из цикла while True
            except Exception as e_loop:
                logger.error(f"Произошла ошибка в главном цикле бота: {e_loop}")
                logger.debug(traceback.format_exc())
                # Пауза перед следующей попыткой, чтобы не заспамить лог или API биржи
                logger.info("Пауза на 60 секунд из-за ошибки в главном цикле...")
                time.sleep(60)

    except Exception as e_global:
        # Этот блок отловит ошибки, возникшие до полной настройки основного логгера
        # или на этапе инициализации компонентов.
        # Используем pre_logger, если он был настроен, или print.
        critical_error_msg = f"Критическая ошибка при запуске или работе бота: {e_global}"
        if temp_logger_setup_done and 'pre_logger' in locals() and pre_logger:
            pre_logger.critical(critical_error_msg)
            pre_logger.debug(traceback.format_exc())
        else: # Если даже pre_logger не настроился
            print(critical_error_msg, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
    finally:
        # Используем основной логгер, если он был инициализирован
        final_log_ref = logging.getLogger("MainBot") if 'logger' in locals() and locals()['logger'] else \
                        (pre_logger if 'pre_logger' in locals() and pre_logger else None)
        if final_log_ref:
            final_log_ref.info("Торговый бот завершил свою работу.")
        else:
            print("Торговый бот завершил свою работу (логгер не был полностью инициализирован).", file=sys.stderr)


# --- Точка входа в приложение ---
if __name__ == '__main__':
    # На случай, если pre_logger не настроился, хоть какой-то вывод
    if not temp_logger_setup_done:
        print("ВНИМАНИЕ: Предварительная настройка логгера не удалась. "
              "Логирование может быть неполным или отсутствовать.", file=sys.stderr)

    run_bot()