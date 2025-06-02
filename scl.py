import numpy as np

code_length = 16
frozen_indices = [0, 1, 2, 3, 4, 5, 8, 9]  # 5G NR
path_limit = 4  # L
tree_depth = int(np.log2(code_length)) + 1


# Функция для добавления шума к сигналу
def awgn(bpsk_signal, error_positions):
    noisy_signal = np.array(bpsk_signal, dtype=float)
    # Переворачиваем BPSK-сигнал на позициях ошибок
    for pos in error_positions:
        noisy_signal[pos] = -noisy_signal[pos]  # 1 -> -1, -1 -> 1
    # Добавляем равномерный шум в диапазоне [-0.75, 0.75] ко всем позициям
    noise = np.random.uniform(-0.75, 0.75, len(bpsk_signal))
    noisy_signal += noise
    # Уменьшаем значения на позициях ошибок в 10 раз, чтобы они были около-пограничные
    for pos in error_positions:
        noisy_signal[pos] /= 10.0
    return noisy_signal


# L функция
def L(left_data, right_data):
    result = np.sign(left_data) * np.sign(right_data) * np.minimum(np.abs(left_data), np.abs(right_data))
    return result.tolist()


# R функция
def R(left_data, right_data, bits):
    result = [right_data[i] + (1 - 2 * bits[1][i]) * left_data[i] for i in range(len(right_data))]
    return result


# Функция для объединения узлов дерева (оно же поднятие наверх, вычисление b)
def b_union(left_node, right_node, left_bits, right_bits):
    metric_sum = left_node[0] + right_node[0]
    merged_bits = [(left_node[1][i] + right_node[1][i]) % 2 for i in range(len(left_node[1]))] + right_node[1]
    combined_list = left_bits + right_bits
    return metric_sum, merged_bits, combined_list


# Функция для "раскидывания" информационных бит по незамороженным листьям и формирование вектора всех листьев
def create_input_vector(info_bits):
    vector = np.zeros(code_length, dtype=int)
    info_positions = [i for i in range(code_length) if i not in frozen_indices]
    for idx, pos in enumerate(info_positions):
        vector[pos] = info_bits[idx]
    return vector


# Функция для кодирования вектора
def polar_encode(u):
    # Вместо "древесного ручного" способа кодирования (с подниманием вверх)
    # я использую домножение вектора u на генерирующую матрицу
    depth = int(np.log2(code_length))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F  # сюда будем собирать генерирующую матрицу
    for i in range(1, depth):  # depth-ая кронекеровская степень
        G = np.kron(G, F)  # кронекеровская степень из лекций <=> фрактал матрицы
    u = np.array(u) % 2
    x = np.mod(np.dot(u, G), 2)
    bpsk_signal = [1 if bit == 0 else -1 for bit in x]
    return bpsk_signal


# Функция для декодирования вектора после шума (y)
def polar_decode(signal, depth=0, node_idx=0):
    # Если это листик (оно же рекурсивный выход в моей программе)
    if depth == tree_depth - 1:
        decisions = []
        decoded_bits = []
        # Если лист не заморожен
        if node_idx not in frozen_indices:
            # Если значение в листе отрицательное
            if signal[0] < 0:
                # Добавляем ещё два решения - развилку на PM дереве
                # Первое - если декодировали правильно, тогда добавляем 0 в PM и бит "1"
                decisions.append((0, [1]))
                decoded_bits.append([1])
                # Второе - если вдруг декодировали неправильно (на самом деле "0"), тогда добавляем l1 в PM и бит "0"
                decisions.append((np.abs(signal[0]), [0]))
                decoded_bits.append([0])
            else:
                # Добавляем ещё два решения - развилку на PM дереве
                # Первое - если декодировали правильно, тогда добавляем 0 в PM и бит "0"
                decisions.append((0, [0]))
                decoded_bits.append([0])
                # Второе - если вдруг декодировали неправильно (на самом деле "1"), тогда добавляем l1 в PM и бит "1"
                decisions.append((np.abs(signal[0]), [1]))
                decoded_bits.append([1])
        # Если лист заморожен
        else:
            # Если значение в листе положительное, то ошибки в замороженном листе нет
            if signal[0] >= 0:
                decisions.append((0, [0]))
                decoded_bits.append([0])
            # Иначе понимаем, что ошибка точно есть - добавляем l1 в PM
            else:
                decisions.append((np.abs(signal[0]), [0]))
                decoded_bits.append([0])
        return decisions, decoded_bits
    # Если это не листик, значит будем рекурсивно что-то делать для потомков
    else:
        # Делим список (<=> вектор <=> массив b-шек) пополам
        half = len(signal) // 2
        left_signal = signal[:half]
        right_signal = signal[half:]
        # L функция для спуска влево. Т.к. питон с нампаем умные, они делают как раз то, что нам нужно,
        # так L функция хорошо отрабатывает около-полиморфизм)))
        left_data = L(left_signal, right_signal)

        # Рекурсия. Всё, что выше - рекурсивный спуск, дальше - подъем.
        # Вызываем её для левого поддерева с глубиной на 1 больше и 2x индексом (левое поддерево в бинарном дереве)
        left_decisions, left_decoded = polar_decode(left_data, depth + 1, 2 * node_idx)

        # R функция для спуска вправо.
        # Проделываем её для каждой "решающей вселенной" после декодирования левого поддерева
        right_data = [R(left_signal, right_signal, decision) for decision in left_decisions]
        # Теперь для каждой "решающей вселенной" надо проделать декодирование правого поддерева
        selection = []
        for i in range(len(right_data)):
            # Вторая рекурсия.
            # Вызываем декодирование каждого правого поддерева для каждой "решающей вселенной" после
            # левого поддерева с аргументами: i-ая вселенная после R функции,
            # длина+1, 2x+1 индекс (правый потомок в двоичном дереве)
            right_decisions, right_decoded = polar_decode(right_data[i], depth + 1, 2 * node_idx + 1)
            # Теперь в каждой паре "вселенной решений" поднимаемся наверх
            for j in range(len(right_decisions)):
                selection.append(b_union(left_decisions[i], right_decisions[j], left_decoded[i], right_decoded[j]))
        # Обрезаем лишние "вселенные решений", оставляем только топовые, для этого сортируем по убыванию PM и обрезаем
        selection = sorted(selection, key=lambda x: x[0])[:path_limit]
        # Формируем ответы для текущего узла. В selection находится: PM, биты для текущего пути, и
        # полная последовательность битов, представляющая один из возможных путей декодирования на текущем уровне дерева
        result_tuples = [(item[0], item[1]) for item in selection]
        result_decoded = [item[2] for item in selection]  # вся декодированная строка на этом моменте
        return result_tuples, result_decoded


# Функция для извлечения декодированных результатов
def extract_results(decision_tuples, decoded_lists):
    # Список для хранения результатов
    extracted_results = []
    # Перебираем все пути декодирования
    for i in range(len(decision_tuples)):
        # Получаем метрику пути
        path_metric = decision_tuples[i][0]
        # Преобразуем декодированный код в строку
        full_code = ''.join(map(str, decoded_lists[i]))
        # Извлекаем информационные биты
        info_code = ""
        for index in range(len(decoded_lists[i])):
            if index not in frozen_indices:
                info_code += str(decoded_lists[i][index])
        # Собираем кортеж из извлечённых значений
        extracted_results.append((path_metric, full_code, info_code))
    return extracted_results


def main(error_indexes, inf_vector=[1, 0, 1, 0, 1, 1, 0, 1]):
    global code_length, frozen_indices, path_limit, tree_depth

    print(f"Max paths: {path_limit}")
    vector = create_input_vector(inf_vector)
    print(f"Вектор бит после раскидывания: {vector}")
    bpsk = polar_encode(vector)
    print(f"После bpsk: {bpsk}")
    noisy_signal = awgn(bpsk, error_indexes)
    print(f'После awgn: {noisy_signal}')

    decision_tuples, decoded_lists = polar_decode(noisy_signal)

    print(f'Передавали биты: {''.join(map(str, inf_vector))}')
    results = extract_results(decision_tuples, decoded_lists)
    print(f"Метрика\tПолный код\t\t\tИнформационный код")
    for metric, full_code, info_code in results:
        print(f"{metric:.2f}\t{full_code}\t{info_code}")


if __name__ == "__main__":
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    # f f f f f f i i f f i  i  i  i  i  i  i
    error_pos_list = [[7, 8],
                      [3, 6, 7],
                      [3, 5, 10],
                      [6, 7, 10],
                      [3, 6, 9, 12],
                      [6, 7, 10, 11]]
    for i in range(len(error_pos_list)):
        print(f'Позиции ошибок на данной итерации: {error_pos_list[i]}')
        main(error_pos_list[i])
        print()

