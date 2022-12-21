import json
import random
import pickle
import cv2
import numpy as np

N_height = 8
M_width = 8
P_neurons = 16
e_mistake = 300
a_learning = 0.0001
image = 'img.png'
compressed = 'compressed_img.bin'

# rewrited
def init_by_zero(size_1, size_2):
    matrix = [[0 for i in range(size_1)] for j in range(size_2)]
    return matrix

def init_matrix(p, n, m):
    matrix = init_by_zero(p, n * m * 3)
    for i in range(n * m * 3):
        for j in range(p):
            matrix[i][j] = random.randint(-1, 1)/100
    #print(matrix)
    return matrix

# rewrited
def transposition(matrix):
    t_matrix = init_by_zero(len(matrix), len(matrix[0]))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            #print(matrix)
            t_matrix[j][i] = matrix[i][j]
    return t_matrix

# rewrited
def matrix_multiplication(m_1, m_2):
    result = init_by_zero(len(m_2[0]), len(m_1))
    for row1 in range(len(m_1)):
        for col2 in range(len(m_2[0])):
            for common in range(len(m_1[0])):
                result[row1][col2] += m_1[row1][common] * m_2[common][col2]
    return result

#rewrited
def delta(m_1, m_2):
    d = init_by_zero(len(m_1[0]), len(m_1))
    for i in range(len(m_1)):
        for j in range(len(m_1[0])):
            d[i][j] = m_1[i][j] - m_2[i][j]
    return d

#rewrited
def alpha_matrix(matrix, alpha):
    a = init_by_zero(len(matrix[0]), len(matrix))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            a[i][j] = matrix[i][j]*alpha
    return a

#rewrited
def sqrt_error(delta_x):
    result = 0
    for i in range(len(delta_x[0])):
        result = result + delta_x[0][0]**2
    return result


def summary_error(e):
    result = 0
    for each_e in range(len(e)):
        result = result + e[each_e]
    return result


def weights_correction(filename, neuro_num, block_width, block_height, error, alpha):
    w_1 = init_matrix(neuro_num, block_width, block_height)
    w_2 = transposition(w_1)
    image = cv2.imread(str(filename))
    img, num_of_blocks = image_to_vector(image, block_width, block_height)
    E = [100000]
    while summary_error(E) > error:
        print(summary_error(E))
        E = []
        for block in range(num_of_blocks):
            X = transposition(img[block])
            Y = (matrix_multiplication(X, w_1))
            result = matrix_multiplication(Y, w_2)
            delta_x = delta(result, X)
            buffer = w_2
            w_2 = delta(w_2, alpha_matrix(matrix_multiplication(transposition(Y), delta_x), alpha))
            w_1 = delta(w_1, alpha_matrix(matrix_multiplication(transposition(X), matrix_multiplication(delta_x, transposition(buffer))), alpha))
            E.append(sqrt_error(delta_x))
    with open('neu_net.json', 'w', encoding="utf-8") as file:
        temp_dict = {"n": block_width, "m": block_height, "neurons": neuro_num, "w1": w_1, "w2": w_2}
        json.dump(temp_dict, file, indent=2)


def compress_image(filename):
    with open('data_600.json', 'r', encoding="utf-8") as file:
        info = json.load(file)
        block_wdth = info['n']
        block_hght = info['m']
        neuro_num = info['neurons']
        W1 = info['w1']
        W2 = info['w2']
    img = cv2.imread(str(filename))
    print(img)
    width, height = img.shape[0], img.shape[1]
    img, num_of_blocks = image_to_vector(img, block_wdth, block_hght)
    compressed_img = []
    for block in range(num_of_blocks):
        X = transposition(img[block])
        Y = (matrix_multiplication(X, W1))
        compressed_img.append(Y)
    with open('compressed_img.bin', "wb") as file:
        pickle.dump([compressed_img, width - width % block_wdth, height - height % block_hght, num_of_blocks], file)


def decompress_image(file):
    with open(file, "rb") as file:
        data = pickle.load(file)
        img = data[0]
        width = data[1]
        height = data[2]
        num_of_blocks = data[3]
    with open('data_600.json', 'r', encoding="utf-8") as file:
        info = json.load(file)
        block_wdh = info['n']
        block_hght = info['m']
        neuro_num = info['neurons']
        W1 = info['w1']
        W2 = info['w2']
    new_img = []
    for block in range(num_of_blocks):
        result = matrix_multiplication(img[block], W2)
        new_img.append(transposition(result))
    vector_to_image(new_img, width, height, block_wdh, block_hght)


def image_to_vector(img, block_wdth, block_hght):
    arr_x = []
    L = (img.shape[0]//block_wdth)*(img.shape[1]//block_hght)
    for i in range(L):
        row = i//(img.shape[1]//block_hght)
        col = i % (img.shape[1]//block_hght)
        X = []
        for j_width in range(row * block_wdth, row * block_wdth + block_wdth):
            for k_height in range(col * block_hght, col * block_hght + block_hght):
                for a_color in range(3):
                    value = (2 * img[j_width][k_height][a_color] / 255) -1
                    X.append([value])
        arr_x.append(X)
    return arr_x, L


def vector_to_image(img, width, height, block_width, block_height):
    arr = np.zeros((width, height, 3), np.uint8)
    x = -1
    for i in range(width // block_height):
        for j in range(height // block_width):
            y = 0
            x += 1
            #понятные индексы, не 1 буква
            for a_height in range(i * block_height, i * block_height + block_height):
                for b_wifth in range(j * block_width, j * block_width + block_width):
                    for z_color in range(3):
                        arr[a_height, b_wifth, z_color] = 255 * (img[x][y][0] + 1) / 2
                        y += 1
    print('Показано')
    cv2.imshow('img', arr)
    cv2.waitKey(0)


if __name__ == '__main__':
    print("1) Обучение\n2) Сжатие\n3) Восстановление\n")
    menu = input()
    if menu == '1':
        weights_correction(image, P_neurons, M_width, N_height, e_mistake, a_learning)
    elif menu == '2':
        compress_image(image)
        print("Готово")
    elif menu == '3':
        decompress_image(compressed)
    elif menu == '4':
        m1 = [[1, 2, 3], [0, 2, 5]]
        m2 = [[1, 2], [0, 2], [1, 1]]
        matrix_multiplication(m1, m2)


