# Алгоритм вычисления контрольной цифры C

# Веса для вычисления state_14 (схема cycle9_1_5)
# d1*1 + d2*2 + ... + d9*9 + d10*1 + d11*2 + d12*3 + d13*4 + d14*5
WEIGHTS_FOR_STATE14 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5]
M_STATE = 50
INITIAL_OFFSET_STATE14 = 0

# Таблицы Lookup_C[state_14][d15] = C
LOOKUP_TABLES_C = {
    0: {4: 4},
    3: {1: 7},
    4: {0: 5, 1: 0, 2: 9, 4: 8, 5: 2, 6: 7, 7: 3, 8: 1, 9: 6},
    7: {0: 0},
    9: {0: 7, 1: 8, 2: 5, 4: 2, 6: 3, 7: 4, 8: 6, 9: 9,
        # СЮДА НУЖНО ДОБАВИТЬ:
        # 3: ЗНАЧЕНИЕ_C_ДЛЯ_STATE9_D15_3,
        # 5: ЗНАЧЕНИЕ_C_ДЛЯ_STATE9_D15_5,
       },
    14: {1: 3, 3: 8, 4: 1, 5: 6, 7: 9, 8: 2, 9: 5},
    15: {0: 3, 4: 0},
    19: {0: 2, 1: 6, 2: 4, 3: 9, 4: 3, 9: 8},
    20: {5: 0, 6: 7, 8: 1, 9: 6},
    24: {0: 8, 2: 0, 3: 1, 5: 3, 6: 9, 7: 6, 8: 5, 9: 4},
    29: {0: 3, 2: 2, 3: 5, 4: 0, 5: 8, 6: 6, 7: 1, 8: 9, 9: 7},
    # СЮДА НУЖНО ДОБАВИТЬ ПРАВИЛО ДЛЯ state_14 = 31:
    # 31: {
    #     0: ЗНАЧЕНИЕ_C_ДЛЯ_STATE31_D15_0,
    #     2: ЗНАЧЕНИЕ_C_ДЛЯ_STATE31_D15_2,
    #     4: ЗНАЧЕНИЕ_C_ДЛЯ_STATE31_D15_4,
    #     5: ЗНАЧЕНИЕ_C_ДЛЯ_STATE31_D15_5,
    #     9: ЗНАЧЕНИЕ_C_ДЛЯ_STATE31_D15_9,
    #     # ... и возможно для других d15, если для state_14=31 есть полная таблица
    # },
    33: {0: 8, 1: 1, 2: 4, 3: 9, 5: 0, 6: 7, 7: 3, 8: 2, 9: 6},
    34: {0: 1, 1: 9, 3: 6, 5: 7, 6: 4, 7: 2, 8: 8, 9: 0,
         # СЮДА НУЖНО ДОБАВИТЬ:
         # 4: ЗНАЧЕНИЕ_C_ДЛЯ_STATE34_D15_4,
        },
    # state_14 = 35 -> Линейный режим W5 (уже есть правило)
    # state_14 = 38 -> Линейный режим W1 (уже есть правило)
    39: {0: 0, 1: 1, 2: 8, 3: 3, 4: 6, 6: 5, 7: 7, 8: 4, 9: 2},
    44: {0: 6, 2: 1, 3: 7, 4: 9, 6: 2, 7: 8, 8: 0, 9: 3},
    49: {0: 9, 1: 7, 2: 6, 3: 2, 5: 5, 6: 8, 7: 0, 8: 3, 9: 1}
}

def calculate_check_digit(payload_d1_to_d15_list):
    if len(payload_d1_to_d15_list) != 15:
        raise ValueError("Полезная нагрузка должна содержать ровно 15 цифр.")
    for digit in payload_d1_to_d15_list:
        if not (0 <= digit <= 9):
            raise ValueError(f"Все цифры в полезной нагрузке должны быть от 0 до 9. Найдено: {digit}")

    payload_d1_to_d14 = payload_d1_to_d15_list[:14]
    d15 = payload_d1_to_d15_list[14]

    s_ctx = INITIAL_OFFSET_STATE14
    for i in range(14):
        s_ctx += payload_d1_to_d14[i] * WEIGHTS_FOR_STATE14[i]
    state_14 = s_ctx % M_STATE

    check_digit_c = -1

    if state_14 == 35:
        check_digit_c = (10 - (4 + 5 * d15) % 10) % 10
    elif state_14 == 38:
        check_digit_c = (10 - (7 + 1 * d15) % 10) % 10
    elif state_14 in LOOKUP_TABLES_C:
        table_for_state14 = LOOKUP_TABLES_C[state_14]
        if d15 in table_for_state14:
            check_digit_c = table_for_state14[d15]
        else:
            print(f"Предупреждение: для state_14={state_14} нет данных для d15={d15} в таблице.")
    else:
        print(f"Предупреждение: для state_14={state_14} нет соответствующей таблицы или правила.")
        
    return check_digit_c

def format_payload_with_hyphens(payload_d1_to_d15_list):
    if len(payload_d1_to_d15_list) != 15:
        return "".join(map(str, payload_d1_to_d15_list))
    
    s = "".join(map(str, payload_d1_to_d15_list))
    sss = s[0:3]
    yy = s[3:5]
    nnnnnnnnnn = s[5:15]
    return f"{sss}-{yy}-{nnnnnnnnnn}"

if __name__ == "__main__":
    print("Программа для вычисления контрольной цифры.")
    print("Формат ввода: 15 цифр без пробелов или дефисов (например, 643230005431990)")
    print("Для выхода введите 'exit'.")

    while True:
        input_str = input("\nВведите 15 цифр полезной нагрузки: ").strip()
        if input_str.lower() == 'exit':
            break

        if len(input_str) != 15 or not input_str.isdigit():
            print("Ошибка: Введите ровно 15 цифр.")
            continue

        payload_list = [int(d) for d in input_str]
        
        # Вычисляем state_14 для информации, даже если C не будет найдена
        s_ctx_info = INITIAL_OFFSET_STATE14
        for i in range(14):
            s_ctx_info += payload_list[i] * WEIGHTS_FOR_STATE14[i]
        calculated_state_14_info = s_ctx_info % M_STATE

        try:
            calculated_c = calculate_check_digit(payload_list)

            print(f"  Вычисленный state_14: {calculated_state_14_info}") # Печатаем state_14 всегда
            if calculated_c != -1:
                full_code_hyphenated = format_payload_with_hyphens(payload_list)
                print(f"  Контрольная цифра (C): {calculated_c}")
                print(f"  Полный код: {full_code_hyphenated}-{calculated_c}")
                print(f"  Полный код (без дефисов): {input_str}{calculated_c}")
            else:
                # Сообщение об ошибке уже выводится из calculate_check_digit
                print("  Не удалось вычислить контрольную цифру для данных входных данных.")
        
        except ValueError as e:
            print(f"Ошибка при обработке: {e}")
        except Exception as e_gen:
            print(f"Произошла непредвиденная ошибка: {e_gen}")

    print("Программа завершена.")