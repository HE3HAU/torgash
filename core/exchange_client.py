import ccxt
import logging
import traceback
import time
import os # Важный импорт для тестового блока
from .utils import safe_float_convert # Используем относительный импорт

class ExchangeClient:
    """
    Клиент для взаимодействия с криптовалютной биржей через ccxt.
    """
    def __init__(self,
                 exchange_name: str,
                 api_key: str,
                 api_secret: str,
                 api_password: str = None, # Для бирж типа OKX, которые требуют passphrase
                 is_testnet: bool = True, # По умолчанию используем testnet
                 logger_instance: logging.Logger = None,
                 default_spot_category: str = 'spot' # Для Bybit v5, например
                 ):
        """
        Инициализирует клиент биржи.

        :param exchange_name: Название биржи (например, 'bybit', 'okx').
        :param api_key: API ключ.
        :param api_secret: API секрет.
        :param api_password: API пароль/passphrase (если требуется биржей).
        :param is_testnet: True для использования тестовой сети, False для реальной.
        :param logger_instance: Экземпляр логгера.
        :param default_spot_category: Категория для спотовых операций (например, 'spot' для Bybit v5).
                                     Это значение будет использовано для параметра 'category' в ордерах.
        """
        self.exchange_name = exchange_name.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_password = api_password
        self.is_testnet = is_testnet
        self.default_spot_category = default_spot_category

        self.logger = logger_instance if logger_instance else logging.getLogger(__name__)
        self.exchange = None

        if not self.api_key or not self.api_secret:
            self.logger.critical(f"API ключ или секрет не предоставлены для биржи {self.exchange_name}. "
                                 "Клиент не сможет выполнять аутентифицированные запросы.")


    def initialize(self) -> bool:
        """
        Инициализирует и настраивает экземпляр ccxt для указанной биржи.
        Загружает рынки.

        :return: True в случае успеха, False в случае ошибки.
        """
        self.logger.info(f"Инициализация клиента для биржи: {self.exchange_name} (Testnet: {self.is_testnet})")
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
        except AttributeError:
            self.logger.error(f"Биржа '{self.exchange_name}' не найдена в ccxt.")
            return False

        config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'options': {
                'defaultType': 'spot', # Общее указание на спот
            },
            'verbose': False,
        }

        if self.api_password:
            config['password'] = self.api_password

        if self.exchange_name == 'bybit':
            config['options']['testnet'] = self.is_testnet
            self.logger.info(f"Настройка Bybit с testnet={self.is_testnet}. "
                             f"Для ордеров будет использоваться категория '{self.default_spot_category}'.")
        elif self.exchange_name == 'okx':
            if self.is_testnet:
                self.logger.info("Включение режима песочницы (демо-торговли) для OKX.")
                config['options']['test'] = self.is_testnet
            self.logger.info(f"Настройка OKX. Testnet/Sandbox режим: {self.is_testnet}")
        else:
            if self.is_testnet and 'test' not in config['options']:
                 config['options']['test'] = True
                 self.logger.info(f"Настройка {self.exchange_name} с общим флагом testnet=True.")

        try:
            self.exchange = exchange_class(config)
            if self.exchange_name == 'okx' and self.is_testnet and hasattr(self.exchange, 'set_sandbox_mode'):
                self.logger.info("OKX: Попытка вызова set_sandbox_mode(True) для тестовой сети.")
                self.exchange.set_sandbox_mode(True)

            self.logger.info(f"Загрузка рынков для {self.exchange_name}...")
            self.exchange.load_markets()
            self.logger.info(f"Рынки для {self.exchange_name} успешно загружены.")

            api_url_display = "N/A"
            if self.exchange.urls.get('api'): # Проверяем наличие ключа 'api'
                if isinstance(self.exchange.urls['api'], dict):
                    api_url_display = self.exchange.urls['api'].get(self.default_spot_category,
                                                                  self.exchange.urls['api'].get('public',
                                                                                                str(self.exchange.urls['api'])))
                    if isinstance(api_url_display, dict): api_url_display = str(api_url_display)
                else:
                    api_url_display = self.exchange.urls['api']
            self.logger.info(f"Используемый API URL (пример): {api_url_display}")
            return True
        except ccxt.AuthenticationError as e:
            self.logger.error(f"Ошибка аутентификации при подключении к {self.exchange_name}: {e}. "
                               "Проверьте правильность API ключей и их прав.")
        except ccxt.NetworkError as e:
            self.logger.error(f"Сетевая ошибка при подключении или загрузке рынков для {self.exchange_name}: {e}")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Ошибка биржи {self.exchange_name} при инициализации: {e}")
        except Exception as e:
            self.logger.error(f"Непредвиденная ошибка при инициализации биржи {self.exchange_name}: {e}")
        
        self.logger.debug(traceback.format_exc())
        self.exchange = None
        return False


    def _get_request_params(self, params: dict = None) -> dict:
        """
        Подготавливает параметры запроса, добавляя категорию для Bybit.
        """
        request_params = params.copy() if params else {}
        if self.exchange_name == 'bybit':
            if 'category' not in request_params:
                # self.default_spot_category должен быть 'spot' для спотовых ордеров,
                # так как мы используем это значение из конструктора
                request_params['category'] = self.default_spot_category
                self.logger.debug(f"Bybit: Установлен параметр 'category': '{self.default_spot_category}' для запроса.")
        return request_params


    def get_current_price(self, pair: str) -> float | None:
        if not self.exchange:
            self.logger.error("Клиент биржи не инициализирован. Невозможно получить цену.")
            return None
        try:
            ticker = self.exchange.fetch_ticker(pair)
            last_price = safe_float_convert(ticker.get('last'))
            if last_price is not None:
                 self.logger.debug(f"Текущая цена для {pair}: {last_price}")
                 return last_price
            else:
                self.logger.warning(f"Не удалось получить 'last' цену из тикера для {pair}. Тикер: {ticker}")
                return None
        except ccxt.NetworkError as e: self.logger.error(f"Сетевая ошибка при получении цены для {pair}: {e}")
        except ccxt.ExchangeError as e: self.logger.error(f"Ошибка биржи при получении цены для {pair}: {e}")
        except Exception as e:
            self.logger.error(f"Непредвиденная ошибка при получении цены для {pair}: {e}")
            self.logger.debug(traceback.format_exc())
        return None


    def get_balance(self, currency_code: str) -> float | None:
        if not self.exchange:
            self.logger.error("Клиент биржи не инициализирован. Невозможно получить баланс.")
            return None
        try:
            fetch_balance_params = {}
            if self.exchange_name == 'bybit':
                # Для Bybit v5 API, если аккаунт Unified Trading Account,
                # accountType должен быть 'UNIFIED' для запроса ОБЩЕГО баланса счета.
                # Отдельные балансы (спот, деривативы) находятся внутри этого ответа.
                fetch_balance_params = {'accountType': 'UNIFIED'} # <--- ИЗМЕНЕНИЕ ЗДЕСЬ
                self.logger.debug(f"Bybit: Используем accountType='UNIFIED' для fetch_balance.")

            balance_data = self.exchange.fetch_balance(params=fetch_balance_params)
            self.logger.debug(f"Ответ fetch_balance для {currency_code} ({self.exchange_name}, params: {fetch_balance_params}): {balance_data}")

            # Пытаемся извлечь баланс через стандартную структуру ccxt (free, used, total)
            # Это может сработать, если ccxt правильно парсит ответ от UNIFIED account для спота
            if currency_code.upper() in balance_data and 'free' in balance_data[currency_code.upper()]:
                free_balance_ccxt = safe_float_convert(balance_data[currency_code.upper()].get('free'))
                if free_balance_ccxt is not None:
                    self.logger.info(f"Баланс {currency_code} (ccxt 'free'): доступно {free_balance_ccxt}")
                    return free_balance_ccxt

            # Если не сработало или для большей точности для Bybit UTA, парсим 'info'
            if self.exchange_name == 'bybit' and \
               'info' in balance_data and isinstance(balance_data['info'], dict) and \
               'result' in balance_data['info'] and isinstance(balance_data['info']['result'], dict) and \
               'list' in balance_data['info']['result'] and isinstance(balance_data['info']['result']['list'], list):
                
                # Ищем данные для счета UNIFIED (обычно он один в списке для UTA)
                unified_account_details = None
                for acc_details_outer in balance_data['info']['result']['list']:
                    # В Bybit v5 ответ fetchBalance для UNIFIED содержит список с одним элементом,
                    # у которого accountType='UNIFIED' (или SPOT, если это старый тип аккаунта).
                    # Нас интересует тот, у которого 'coin' - список монет.
                    if isinstance(acc_details_outer, dict) and acc_details_outer.get('accountType') == 'UNIFIED' and 'coin' in acc_details_outer:
                         unified_account_details = acc_details_outer
                         break
                    # Запасной вариант, если тип не UNIFIED, но есть 'coin' (может быть SPOT аккаунт)
                    elif isinstance(acc_details_outer, dict) and 'coin' in acc_details_outer and not unified_account_details:
                        unified_account_details = acc_details_outer


                if unified_account_details and 'coin' in unified_account_details and isinstance(unified_account_details['coin'], list):
                    for coin_data in unified_account_details['coin']:
                        if isinstance(coin_data, dict) and coin_data.get('coin') == currency_code.upper():
                            # Для спотовой торговли в UTA важен 'availableBalance' или 'availableToWithdraw'
                            # 'walletBalance' = total, 'locked' = in orders/positions
                            available_bal_str = coin_data.get('availableBalance', coin_data.get('availableToWithdraw'))
                            
                            available_bal = None
                            if available_bal_str is not None:
                                available_bal = safe_float_convert(available_bal_str)
                            else: # Fallback, если нет 'availableBalance'/'availableToWithdraw'
                                wallet_balance = safe_float_convert(coin_data.get('walletBalance'))
                                locked = safe_float_convert(coin_data.get('locked', 0.0)) # locked может отсутствовать или быть 0
                                if wallet_balance is not None: # locked может быть None, если 0
                                    available_bal = wallet_balance - (locked if locked is not None else 0.0)
                            
                            if available_bal is not None:
                                self.logger.info(f"Баланс {currency_code} (Bybit UTA '{unified_account_details.get('accountType')}'): доступно {available_bal}")
                                return available_bal
                    self.logger.warning(f"Валюта {currency_code} не найдена в списке монет ('coin') аккаунта Bybit UTA: {unified_account_details.get('accountType')}.")
                else:
                    self.logger.warning(f"Не найдена ожидаемая структура ('coin' list) в деталях аккаунта Bybit UTA. "
                                       f"Проверьте структуру `balance_data['info']['result']['list']`.")

            self.logger.warning(f"Не удалось определить баланс для {currency_code} из ответа биржи {self.exchange_name}. "
                                f"Возвращаем 0.0. Проверьте структуру ответа в DEBUG логах.")
            return 0.0 # Возвращаем 0.0, если валюта не найдена или баланс не определен, но не было ошибки запроса

        except ccxt.NetworkError as e: self.logger.error(f"Сетевая ошибка при получении баланса {currency_code}: {e}")
        except ccxt.ExchangeError as e: self.logger.error(f"Ошибка биржи при получении баланса {currency_code}: {e}")
        except Exception as e:
            self.logger.error(f"Непредвиденная ошибка при получении баланса {currency_code}: {e}")
            self.logger.debug(traceback.format_exc())
        return None # Возвращаем None в случае ошибки самого запроса

    def create_limit_order(self, pair: str, side: str, amount: float, price: float, params: dict = None) -> dict | None:
        if not self.exchange:
            self.logger.error("Клиент биржи не инициализирован. Невозможно создать ордер.")
            return None
        
        final_params = self._get_request_params(params)
        try:
            base_currency = pair.split('/')[0]
            quote_currency = pair.split('/')[1]
            self.logger.info(f"Попытка создать {side} ордер: {amount:.8f} {base_currency} для {pair} по цене {price:.4f} {quote_currency} с params: {final_params}")
            
            order = self.exchange.create_limit_order(pair, side, amount, price, final_params)
            self.logger.info(f"Успешно создан {side} ордер ID: {order.get('id')} для {pair} на {self.exchange_name}")
            return order
        except ccxt.InsufficientFunds as e:
            self.logger.error(f"Недостаточно средств для создания ордера ({side} {amount} {pair} @ {price}): {e}")
        except ccxt.InvalidOrder as e:
            self.logger.error(f"Невалидный ордер ({side} {amount} {pair} @ {price}): {e}. Проверьте мин. размер ордера и точность (precision).")
        except ccxt.NetworkError as e: self.logger.error(f"Сетевая ошибка при создании ордера: {e}")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Ошибка биржи при создании ордера ({side} {amount} {pair} @ {price}): {e}")
        except Exception as e:
            self.logger.error(f"Непредвиденная ошибка при создании ордера: {e}")
            self.logger.debug(traceback.format_exc())
        return None

    def get_order_status_by_id(self, order_id: str, pair: str, params: dict = None) -> str | None:
        if not self.exchange:
            self.logger.error("Клиент биржи не инициализирован. Невозможно получить статус ордера.")
            return None

        final_params = self._get_request_params(params)
        self.logger.debug(f"Запрос статуса ордера ID: {order_id} для пары {pair} с params: {final_params}")

        try:
            if self.exchange.has['fetchOrder']:
                order = self.exchange.fetch_order(order_id, pair, params=final_params)
                self.logger.debug(f"Статус ордера {order_id} (fetchOrder): {order.get('status')}. Детали: {order}")
                return order.get('status')
            
            self.logger.warning(f"Метод fetchOrder не поддерживается биржей {self.exchange_name} или не вернул результат. "
                               f"Попытка найти ордер через fetchOpenOrders/fetchClosedOrders.")
            if self.exchange.has['fetchOpenOrders']:
                open_orders = self.exchange.fetch_open_orders(pair, params=final_params)
                for o in open_orders:
                    if o['id'] == order_id:
                        self.logger.debug(f"Ордер {order_id} найден среди открытых. Статус: {o['status']}")
                        return o['status']
            
            if self.exchange.has['fetchClosedOrders']:
                closed_params = final_params.copy()
                closed_params.update({'limit': 100}) 
                closed_orders = self.exchange.fetch_closed_orders(pair, params=closed_params)
                for o in closed_orders:
                    if o['id'] == order_id:
                        self.logger.debug(f"Ордер {order_id} найден среди закрытых. Статус: {o['status']}")
                        return o['status']

            self.logger.warning(f"Ордер {order_id} не найден для пары {pair} после всех проверок.")
            return 'not_found_or_too_old'

        except ccxt.OrderNotFound:
            self.logger.info(f"Ордер {order_id} не найден на бирже {self.exchange_name} для пары {pair} (OrderNotFound).")
            return 'not_found_or_too_old'
        except ccxt.NetworkError as e: self.logger.error(f"Сетевая ошибка при получении статуса ордера {order_id}: {e}")
        except ccxt.ExchangeError as e: self.logger.error(f"Ошибка биржи при получении статуса ордера {order_id}: {e}")
        except Exception as e:
            self.logger.error(f"Непредвиденная ошибка при получении статуса ордера {order_id}: {e}")
            self.logger.debug(traceback.format_exc())
        return None


    def cancel_order_by_id(self, order_id: str, pair: str, params: dict = None) -> bool:
        if not self.exchange:
            self.logger.error("Клиент биржи не инициализирован. Невозможно отменить ордер.")
            return False
        
        final_params = self._get_request_params(params)
        self.logger.info(f"Попытка отменить ордер ID: {order_id} для пары {pair} с params: {final_params}")
        try:
            if not self.exchange.has['cancelOrder']:
                self.logger.error(f"Биржа {self.exchange_name} не поддерживает отмену ордеров через ccxt (cancelOrder).")
                return False
            
            cancellation_response = self.exchange.cancel_order(order_id, pair, params=final_params)
            self.logger.info(f"Запрос на отмену ордера {order_id} отправлен. Ответ биржи: {cancellation_response}")
            return True
        except ccxt.OrderNotFound:
            self.logger.info(f"Не удалось отменить ордер {order_id}: ордер не найден (возможно, уже исполнен или отменен). Считаем это успехом.")
            return True
        except ccxt.InvalidOrder as e: 
             self.logger.info(f"Не удалось отменить ордер {order_id} (InvalidOrder): {e}. Возможно, ордер уже неактивен.")
             current_status = self.get_order_status_by_id(order_id, pair)
             if current_status in ['closed', 'canceled', 'not_found_or_too_old']:
                 self.logger.info(f"Ордер {order_id} уже в состоянии '{current_status}'. Отмена не требуется.")
                 return True
             return False
        except ccxt.NetworkError as e: self.logger.error(f"Сетевая ошибка при отмене ордера {order_id}: {e}")
        except ccxt.ExchangeError as e: self.logger.error(f"Ошибка биржи при отмене ордера {order_id}: {e}")
        except Exception as e:
            self.logger.error(f"Непредвиденная ошибка при отмене ордера {order_id}: {e}")
            self.logger.debug(traceback.format_exc())
        return False


# --- Тестовый блок (if __name__ == '__main__') ---
if __name__ == '__main__':
    test_logger = logging.getLogger("TestExchangeClient")
    test_logger.setLevel(logging.DEBUG) # DEBUG для подробных логов
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    if not test_logger.hasHandlers(): # Предотвращаем дублирование
        test_logger.addHandler(console_handler)
    
    from dotenv import load_dotenv # Импорт здесь, т.к. нужен только для теста
    # Ищем .env в текущей папке или на уровень выше (если запускаем из папки core)
    env_path_1 = os.path.join(os.getcwd(), '.env')
    env_path_2 = os.path.join(os.path.dirname(os.getcwd()), '.env') # Если запускаем из core/
    env_path_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env') # Если tests/core/client_test.py

    dotenv_file_to_load = None
    for p in [env_path_1, env_path_2, env_path_root]:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            dotenv_file_to_load = abs_p
            break
    
    if dotenv_file_to_load:
        load_dotenv(dotenv_file_to_load)
        test_logger.info(f"Загружены переменные из {dotenv_file_to_load}")
    else:
        test_logger.warning(".env файл не найден в стандартных расположениях. Тесты с аутентификацией могут не работать.")

    BYBIT_API_KEY_TEST = os.getenv("BYBIT_TESTNET_API_KEY")
    BYBIT_API_SECRET_TEST = os.getenv("BYBIT_TESTNET_API_SECRET")

    RUN_BYBIT_TESTS = False # Установите True для тестов с реальными запросами к тестовой сети Bybit
    TEST_PAIR_BYBIT = 'BTC/USDT' 
    TEST_ORDER_AMOUNT_BYBIT = 0.00011 

    if not BYBIT_API_KEY_TEST or not BYBIT_API_SECRET_TEST:
        test_logger.warning("API ключи Bybit Testnet не найдены в .env. Пропуск тестов Bybit, требующих аутентификации.")
        RUN_BYBIT_TESTS = False

    if RUN_BYBIT_TESTS:
        test_logger.info("\n" + "="*10 + " Тестирование Bybit клиента " + "="*10)
        # Убедитесь, что default_spot_category='spot' для спотовой торговли
        bybit_client = ExchangeClient(
            exchange_name='bybit',
            api_key=BYBIT_API_KEY_TEST,
            api_secret=BYBIT_API_SECRET_TEST,
            is_testnet=True,
            logger_instance=test_logger,
            default_spot_category='spot' 
        )

        if bybit_client.initialize():
            test_logger.info("Клиент Bybit успешно инициализирован.")

            price = bybit_client.get_current_price(TEST_PAIR_BYBIT)
            test_logger.info(f"Текущая цена {TEST_PAIR_BYBIT}: {price}")

            usdt_balance = bybit_client.get_balance('USDT')
            btc_balance = bybit_client.get_balance('BTC')
            test_logger.info(f"Баланс USDT: {usdt_balance}")
            test_logger.info(f"Баланс BTC: {btc_balance}")

            if price and usdt_balance is not None and usdt_balance > (price * TEST_ORDER_AMOUNT_BYBIT * 1.1): # *1.1 для запаса
                buy_price_test = round(price * 0.95, 1) # Округляем цену для Bybit (например, до 1 знака для BTC/USDT)
                test_logger.info(f"Попытка создать BUY ордер на {TEST_ORDER_AMOUNT_BYBIT} {TEST_PAIR_BYBIT.split('/')[0]} по цене {buy_price_test}")
                
                buy_order_info = bybit_client.create_limit_order(
                    TEST_PAIR_BYBIT, 'buy', TEST_ORDER_AMOUNT_BYBIT, buy_price_test
                )
                created_buy_order_id = None
                if buy_order_info and 'id' in buy_order_info:
                    created_buy_order_id = buy_order_info['id']
                    test_logger.info(f"Создан BUY ордер ID: {created_buy_order_id}, статус: {buy_order_info.get('status')}")

                    time.sleep(3) # Пауза, чтобы ордер обработался на бирже
                    status = bybit_client.get_order_status_by_id(created_buy_order_id, TEST_PAIR_BYBIT)
                    test_logger.info(f"Статус ордера {created_buy_order_id} (после создания): {status}")

                    if status == 'open': 
                        cancelled = bybit_client.cancel_order_by_id(created_buy_order_id, TEST_PAIR_BYBIT)
                        test_logger.info(f"Результат отмены ордера {created_buy_order_id}: {cancelled}")
                        if cancelled:
                            time.sleep(3)
                            status_after_cancel = bybit_client.get_order_status_by_id(created_buy_order_id, TEST_PAIR_BYBIT)
                            test_logger.info(f"Статус ордера {created_buy_order_id} (после отмены): {status_after_cancel}")
                    else:
                        test_logger.info(f"Ордер {created_buy_order_id} не в статусе 'open' ({status}), отмена не будет протестирована.")
                else:
                    test_logger.error("Не удалось создать тестовый BUY ордер.")
            else:
                test_logger.warning(f"Недостаточно USDT ({usdt_balance}) или не удалось получить цену для создания тестового BUY ордера "
                                   f"(требуется > {price * TEST_ORDER_AMOUNT_BYBIT * 1.1 if price else 'N/A'}).")
        else:
            test_logger.error("Не удалось инициализировать клиент Bybit.")
    else:
        test_logger.info("Пропуск тестов Bybit (RUN_BYBIT_TESTS=False или нет ключей/недостаточно средств).")

    test_logger.info("Тестирование ExchangeClient завершено.")