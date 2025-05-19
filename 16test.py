import os
from collections import defaultdict

# --- Функции, которые нам все еще нужны ---
def parse_payload_check_from_string(code_str):
    """Универсальный парсер для строк с дефисами или без."""
    s = code_str.replace('-', '')
    if len(s) != 16:
        raise ValueError(f"Неверная длина кода (ожидалось 16 цифр): {len(s)} для '{code_str}'")
    if not s.isdigit():
        raise ValueError(f"Код содержит нецифровые символы: '{code_str}'")
        
    payload = [int(d) for d in s[:-1]]
    check = int(s[-1])
    return payload, check

# --- Основная часть ---
if __name__ == "__main__":
    initial_examples_str = [
        "643-23-0005431990-6",
        "643-23-0005431991-1",
        "643-23-0005432000-0",
    ]

    uins_filename = "uins_100.txt" 
    uins_file_content_list = []
    if os.path.exists(uins_filename):
        with open(uins_filename, 'r') as f:
            uins_file_content_list = [line.strip() for line in f if line.strip()]
        print(f"Загружено {len(uins_file_content_list)} строк из {uins_filename}")
    else:
        print(f"Файл {uins_filename} не найден. Будут использованы только initial_examples.")

    all_codes_str_list = initial_examples_str + uins_file_content_list
    unique_codes_str_list = sorted(list(set(filter(None, all_codes_str_list))))
    print(f"Всего уникальных кодов для анализа: {len(unique_codes_str_list)}")

    # 1. ПАРСИНГ ВСЕХ КОДОВ
    parsed_codes_for_analysis = []
    print("\n--- Парсинг кодов ---")
    for code_str in unique_codes_str_list:
        try:
            payload_digits, expected_c = parse_payload_check_from_string(code_str)
            parsed_codes_for_analysis.append({
                'str': code_str,
                'payload': payload_digits,
                'check': expected_c
            })
        except ValueError as e:
            print(f"Ошибка парсинга кода '{code_str}': {e}")
    print(f"Успешно разобрано {len(parsed_codes_for_analysis)} кодов.")
    if not parsed_codes_for_analysis:
        print("Нет валидных кодов для анализа. Завершение.")
        exit()

    # 2. КЛАССИФИКАЦИЯ РЕЖИМОВ ОБРАБОТКИ d15 ДЛЯ КАЖДОГО УНИКАЛЬНОГО ПРЕФИКСА d1..d14
    print("\n--- Классификация режимов d15 для префиксов d1..d14 ---")
    
    prefixes_d1_d14_to_entries = defaultdict(list)
    for code_data in parsed_codes_for_analysis:
        prefix_tuple = tuple(code_data['payload'][:14])
        prefixes_d1_d14_to_entries[prefix_tuple].append({
            'd15': code_data['payload'][14],
            'C': code_data['check'],
            'str': code_data['str']
        })

    prefix_to_d15_mode = {} 
    inv_map_wcand = {1:1, 3:7, 7:3, 9:9}

    for prefix_tuple, entries_for_prefix in prefixes_d1_d14_to_entries.items():
        unique_d15_c_pairs_for_prefix_calc = []
        temp_d15_map_for_wcand = {} 
        for entry in sorted(entries_for_prefix, key=lambda x: x['d15']):
            if entry['d15'] not in temp_d15_map_for_wcand:
                temp_d15_map_for_wcand[entry['d15']] = entry['C']
        
        for d15_val, c_val in temp_d15_map_for_wcand.items():
            unique_d15_c_pairs_for_prefix_calc.append({'d15': d15_val, 'C': c_val})
        
        current_prefix_wcands = set()
        current_prefix_has_undef = False
        
        if len(unique_d15_c_pairs_for_prefix_calc) >= 2:
            for i in range(len(unique_d15_c_pairs_for_prefix_calc)):
                for j in range(i + 1, len(unique_d15_c_pairs_for_prefix_calc)):
                    p1 = unique_d15_c_pairs_for_prefix_calc[i]
                    p2 = unique_d15_c_pairs_for_prefix_calc[j]
                    d_diff = (p1['d15'] - p2['d15'] + 10) % 10
                    c_diff = (p2['C'] - p1['C'] + 10) % 10
                    if d_diff == 0: continue
                    if d_diff in inv_map_wcand:
                        wc = (c_diff * inv_map_wcand[d_diff]) % 10
                        current_prefix_wcands.add(wc)
                    else:
                        current_prefix_has_undef = True
        
        mode_str = "NL_UNDETERMINED"
        map_d15_to_set_of_c = defaultdict(set)
        for entry in entries_for_prefix:
            map_d15_to_set_of_c[entry['d15']].add(entry['C'])
        is_complex_mapping = any(len(c_set) > 1 for c_set in map_d15_to_set_of_c.values())

        if is_complex_mapping:
             mode_str = "NL_MAP_COMPLEX" # d15 -> множество C для этого префикса
        elif len(current_prefix_wcands) == 1 and not current_prefix_has_undef:
            mode_str = f"W{list(current_prefix_wcands)[0]}"
        elif current_prefix_wcands:
            sorted_w = sorted(list(current_prefix_wcands))
            mode_str = f"NL_Wcand_{'_'.join(map(str, sorted_w))}"
            if current_prefix_has_undef: mode_str += "_undef"
        elif current_prefix_has_undef:
             mode_str = "NL_ONLY_UNDEF"
        elif len(unique_d15_c_pairs_for_prefix_calc) == 1:
             mode_str = f"NL_SINGLE_D15_C_{unique_d15_c_pairs_for_prefix_calc[0]['d15']}_{unique_d15_c_pairs_for_prefix_calc[0]['C']}"
        elif not unique_d15_c_pairs_for_prefix_calc : # entries_for_prefix было, но уникальных пар нет (все d15 одинаковые)
             # Это значит, что для этого префикса все записи имеют одно и то же d15.
             # Режим не определен через Wcand, но он может быть простым отображением.
             # Если все C одинаковы, то это NL_SINGLE_D15_C_...
             if len(map_d15_to_set_of_c) == 1: # Только одно значение d15
                 d15_single_val = list(map_d15_to_set_of_c.keys())[0]
                 c_set_single_d15 = list(map_d15_to_set_of_c.values())[0]
                 if len(c_set_single_d15) == 1:
                     mode_str = f"NL_CONST_D15_{d15_single_val}_C_{list(c_set_single_d15)[0]}"
                 else: # одно d15, но разные C - это тоже комплексная карта
                     mode_str = "NL_MAP_COMPLEX"


        prefix_to_d15_mode[prefix_tuple] = mode_str

    print("  Карта 'Префикс -> Режим обработки d15' (классифицировано):")
    for p_tuple_key, m_str_val in prefix_to_d15_mode.items():
         p_str_short = "".join(map(str, p_tuple_key[:5])) + "..." + "".join(map(str,p_tuple_key[-4:]))
         print(f"    {p_str_short} ({len(prefixes_d1_d14_to_entries[p_tuple_key])} экз.) -> {m_str_val}")
    print(f"  Всего классифицировано режимов для {len(prefix_to_d15_mode)} уникальных префиксов d1..d14.")
    if len(prefix_to_d15_mode) == 0 :
        print("Не удалось классифицировать режимы для префиксов.")
        # exit()

    # 3. АНАЛИЗ ПО ИДЕЕ 14.1: Взвешенная сумма d1..d14 как контекст
    print("\n--- Анализ по Идее 14.1: Взвешенная сумма d1..d14 как контекст ---")
    
    pattern_primes_val_for_scheme = [1,3,7,9] # Определяем здесь, чтобы не было ошибки в lambda для словаря
    weight_schemes = {
        "all_1": [1]*14,
        "1_3_alt_evenidx_1": [1 if i%2==0 else 3 for i in range(14)],
        "3_1_alt_evenidx_3": [3 if i%2==0 else 1 for i in range(14)],
        "1_2_alt_evenidx_1": [1 if i%2==0 else 2 for i in range(14)],
        "2_1_alt_evenidx_2": [2 if i%2==0 else 1 for i in range(14)],
        "cycle9_1_5": [(i%9)+1 for i in range(14)], 
        "cycle7_all": [(i%7)+1 for i in range(14)], 
        "primes_1379_cycle": [pattern_primes_val_for_scheme[i % len(pattern_primes_val_for_scheme)] for i in range(14)],
        "rev_cycle9_5_1": [(((13-i)%9)+1) for i in range(14)] 
    }

    M_state_values = [10, 11, 13, 16, 17, 19, 20, 23, 29, 30, 31, 37, 40, 41, 43, 47, 50, 60, 100] 
    initial_offset_values = [0] 

    best_overall_quality = -1 
    best_params_overall = {}

    for ws_name, ws_values in weight_schemes.items():
        for M_state in M_state_values:
            for initial_offset in initial_offset_values:
                state14_to_modes_map = defaultdict(set)
                for p_tuple, d15_mode_for_p in prefix_to_d15_mode.items():
                    payload_d1_d14_current = list(p_tuple) 
                    s14_calc = initial_offset
                    for i in range(14):
                        s14_calc += payload_d1_d14_current[i] * ws_values[i]
                    state14_val_calc = s14_calc % M_state
                    state14_to_modes_map[state14_val_calc].add(d15_mode_for_p)
                
                num_pure_states_calc = 0
                num_total_states_with_data_calc = 0
                num_distinct_modes_in_pure_states_set = set()

                for state_val_map, modes_set_map in state14_to_modes_map.items():
                    if modes_set_map: 
                        num_total_states_with_data_calc +=1
                        if len(modes_set_map) == 1:
                            num_pure_states_calc += 1
                            num_distinct_modes_in_pure_states_set.add(list(modes_set_map)[0]) 
                
                purity_score = 0
                if num_total_states_with_data_calc > 0:
                    purity_score = num_pure_states_calc / num_total_states_with_data_calc
                
                current_quality = purity_score * len(num_distinct_modes_in_pure_states_set)

                if current_quality > best_overall_quality: # Строгое улучшение
                    best_overall_quality = current_quality
                    best_params_overall = {
                        'ws_name': ws_name, 'M_state': M_state, 'offset': initial_offset,
                        'pure_states': num_pure_states_calc, 
                        'total_states_w_data': num_total_states_with_data_calc,
                        'purity_score': purity_score,
                        'distinct_modes_in_pure': len(num_distinct_modes_in_pure_states_set),
                        'map': dict(state14_to_modes_map) 
                    }
                    print(f"    Новая лучшая комбинация: Качество={current_quality:.2f} (Чистота={purity_score:.2f}, Чистых состояний={num_pure_states_calc}/{num_total_states_with_data_calc}, Разных режимов в чистых={len(num_distinct_modes_in_pure_states_set)})")
                    print(f"      Параметры: Веса={ws_name}, M_state={M_state}, Offset={initial_offset}")
                # Можно добавить условие для вывода, если качество такое же, но другие параметры лучше (например, меньше M_state)
                # elif current_quality == best_overall_quality and best_params_overall and M_state < best_params_overall.get('M_state', float('inf')):
                #    # ... (логика обновления, если хотим предпочесть меньший M_state при том же качестве) ...
                #    pass


    if best_params_overall:
        print("\n  --- Лучшая найденная комбинация для state14 и режимов d15 ---")
        bpo = best_params_overall
        print(f"  Параметры: Веса={bpo['ws_name']}, M_state={bpo['M_state']}, Смещение={bpo['offset']}")
        print(f"  Качество={best_overall_quality:.2f} (Чистота={bpo['purity_score']:.2f}, Чистых состояний={bpo['pure_states']}/{bpo['total_states_w_data']}, Разных режимов в чистых={bpo['distinct_modes_in_pure']})")
        
        print("\n  --- Детальная карта state14 -> (режим d15) -> [карта d15:C] для лучшей комбинации ---")
        
        best_ws_values = weight_schemes[bpo['ws_name']]
        best_M_state = bpo['M_state']
        best_initial_offset = bpo['offset']

        state14_to_full_entries_map = defaultdict(list)
        for code_data in parsed_codes_for_analysis:
            payload_d1_d14 = code_data['payload'][:14]
            s14_calc = best_initial_offset
            for i in range(14):
                s14_calc += payload_d1_d14[i] * best_ws_values[i]
            state14_val_calc = s14_calc % best_M_state
            
            state14_to_full_entries_map[state14_val_calc].append({
                'd15': code_data['payload'][14],
                'C': code_data['check'],
                'str': code_data['str']
            })

        for state_val_report, modes_set_report in sorted(bpo['map'].items()):
            if not modes_set_report: continue

            print(f"\n  Для state14 = {state_val_report} (Режим(ы) из классификации: {modes_set_report}):")
            
            entries_for_this_state = state14_to_full_entries_map.get(state_val_report, [])
            if not entries_for_this_state:
                print("    Нет данных для построения карты d15->C для этого state14.")
                continue

            map_d15_to_c_for_state = defaultdict(set)
            for entry in entries_for_this_state:
                map_d15_to_c_for_state[entry['d15']].add(entry['C'])
            
            if not map_d15_to_c_for_state:
                print("    Карта d15->C пуста.")
            
            all_single_c_for_this_state = True
            for d15_key, c_set_val in sorted(map_d15_to_c_for_state.items()):
                print(f"    d15 = {d15_key} -> C = {sorted(list(c_set_val))}")
                if len(c_set_val) > 1:
                    all_single_c_for_this_state = False
            
            if not all_single_c_for_this_state:
                print(f"    ПРЕДУПРЕЖДЕНИЕ: Для state14={state_val_report}, некоторые d15 отображаются в несколько C! Это может указывать на то, что state_14 все еще недостаточно специфичен или есть ошибки в данных/классификации режимов.")
            
            # Специальный анализ для известных нам "чистых" режимов W5 и W1, если они попали сюда
            # (Предполагаем, что 'W5' и 'W1' - это уникальные строки режима)
            if 'W5' in modes_set_report : 
                print("    Анализ для state14 (режим W5): C = (10-(4 + 5*d15)%10)%10")
                for d15_val_check in range(10):
                    c_expected = (10 - (4 + 5 * d15_val_check) % 10) % 10
                    c_actual_set = map_d15_to_c_for_state.get(d15_val_check)
                    if c_actual_set is not None:
                        print(f"      d15={d15_val_check}: Ожидаемый C={c_expected}, Наблюдаемый C={sorted(list(c_actual_set))}")
            
            if 'W1' in modes_set_report: 
                print("    Анализ для state14 (режим W1): C = (10-(7 + 1*d15)%10)%10")
                for d15_val_check in range(10):
                    c_expected = (10 - (7 + 1 * d15_val_check) % 10) % 10
                    c_actual_set = map_d15_to_c_for_state.get(d15_val_check)
                    if c_actual_set is not None:
                        print(f"      d15={d15_val_check}: Ожидаемый C={c_expected}, Наблюдаемый C={sorted(list(c_actual_set))}")
    else:
        print("  Не удалось найти значимой комбинации для state14 с выбранными схемами весов.")

    print("\n--- Анализ завершен ---")