#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IMAGE_SIZE 28 * 28
#define TRAIN_IMAGES 60000
#define TEST_IMAGES 10000
#define INPUT_SIZE IMAGE_SIZE
#define HIDDEN_SIZE1 128
#define HIDDEN_SIZE2 64
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001
#define BATCH_SIZE 32
#define EPOCHS 50

// Функция для чтения изображений
void read_images(const char *filename, unsigned char *images, int num_images) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Ошибка открытия файла");
        exit(1);
    }

    // Чтение заголовка
    int magic_number, num_imgs, rows, cols;
    if (fread(&magic_number, sizeof(int), 1, file) != 1 ||
        fread(&num_imgs, sizeof(int), 1, file) != 1 ||
        fread(&rows, sizeof(int), 1, file) != 1 ||
        fread(&cols, sizeof(int), 1, file) != 1) {
        perror("Ошибка чтения заголовка файла");
        fclose(file);
        exit(1);
    }

    // Преобразование из big-endian в little-endian
    magic_number = __builtin_bswap32(magic_number);
    num_imgs = __builtin_bswap32(num_imgs);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    // Проверка корректности заголовка
    if (magic_number != 2051) {
        fprintf(stderr, "Неправильный магический номер файла изображений\n");
        fclose(file);
        exit(1);
    }
    if (num_imgs != num_images) {
        fprintf(stderr, "Ожидалось %d изображений, но в файле %d\n", num_images, num_imgs);
        fclose(file);
        exit(1);
    }
    if (rows != 28 || cols != 28) {
        fprintf(stderr, "Ожидались изображения 28x28, но в файле %dx%d\n", rows, cols);
        fclose(file);
        exit(1);
    }

    // Чтение данных
    if (fread(images, sizeof(unsigned char), IMAGE_SIZE * num_images, file) != IMAGE_SIZE * num_images) {
        perror("Ошибка чтения данных изображений");
        fclose(file);
        exit(1);
    }
    fclose(file);
}

// Функция для чтения меток
void read_labels(const char *filename, unsigned char *labels, int num_labels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Ошибка открытия файла");
        exit(1);
    }

    // Чтение заголовка
    int magic_number, num_items;
    if (fread(&magic_number, sizeof(int), 1, file) != 1 ||
        fread(&num_items, sizeof(int), 1, file) != 1) {
        perror("Ошибка чтения заголовка файла");
        fclose(file);
        exit(1);
    }

    // Преобразование из big-endian в little-endian
    magic_number = __builtin_bswap32(magic_number);
    num_items = __builtin_bswap32(num_items);

    // Проверка корректности заголовка
    if (magic_number != 2049) {
        fprintf(stderr, "Неправильный магический номер файла меток\n");
        fclose(file);
        exit(1);
    }
    if (num_items != num_labels) {
        fprintf(stderr, "Ожидалось %d меток, но в файле %d\n", num_labels, num_items);
        fclose(file);
        exit(1);
    }

    // Чтение данных
    if (fread(labels, sizeof(unsigned char), num_labels, file) != num_labels) {
        perror("Ошибка чтения данных меток");
        fclose(file);
        exit(1);
    }
    fclose(file);
}

// Функция активации ReLU
double relu(double x) {
    return x > 0 ? x : 0;
}

// Производная ReLU
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Функция активации (сигмоида)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Производная сигмоиды
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Инициализация весов He
void initialize_weights(double *weights, int size, int n) {
    double stddev = sqrt(2.0 / n);
    for (int i = 0; i < size; i++) {
        weights[i] = ((double)rand() / RAND_MAX) * 2 * stddev - stddev;
    }
}

// Прямой проход (forward pass) с ReLU
void forward_pass(double *input, double *hidden1, double *hidden2, double *output, double *w1, double *w2, double *w3) {
    for (int i = 0; i < HIDDEN_SIZE1; i++) {
        hidden1[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden1[i] += input[j] * w1[j * HIDDEN_SIZE1 + i];
        }
        hidden1[i] = relu(hidden1[i]);
    }

    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        hidden2[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE1; j++) {
            hidden2[i] += hidden1[j] * w2[j * HIDDEN_SIZE2 + i];
        }
        hidden2[i] = relu(hidden2[i]);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            output[i] += hidden2[j] * w3[j * OUTPUT_SIZE + i];
        }
        output[i] = sigmoid(output[i]);
    }
}

// Обратное распространение ошибки (backpropagation) с ReLU
void backpropagation(double *input, double *hidden1, double *hidden2, double *output, double *target, double *w1, double *w2, double *w3) {
    double output_error[OUTPUT_SIZE];
    double hidden2_error[HIDDEN_SIZE2];
    double hidden1_error[HIDDEN_SIZE1];

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error[i] = (target[i] - output[i]) * sigmoid_derivative(output[i]);
    }

    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        hidden2_error[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden2_error[i] += output_error[j] * w3[i * OUTPUT_SIZE + j];
        }
        hidden2_error[i] *= relu_derivative(hidden2[i]);
    }

    for (int i = 0; i < HIDDEN_SIZE1; i++) {
        hidden1_error[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            hidden1_error[i] += hidden2_error[j] * w2[i * HIDDEN_SIZE2 + j];
        }
        hidden1_error[i] *= relu_derivative(hidden1[i]);
    }

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE1; j++) {
            w1[i * HIDDEN_SIZE1 + j] += LEARNING_RATE * hidden1_error[j] * input[i];
        }
    }

    for (int i = 0; i < HIDDEN_SIZE1; i++) {
        for (int j = 0; j < HIDDEN_SIZE2; j++) {
            w2[i * HIDDEN_SIZE2 + j] += LEARNING_RATE * hidden2_error[j] * hidden1[i];
        }
    }

    for (int i = 0; i < HIDDEN_SIZE2; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            w3[i * OUTPUT_SIZE + j] += LEARNING_RATE * output_error[j] * hidden2[i];
        }
    }
}

int main() {
    srand(time(NULL));

    // Выделение памяти для данных
    unsigned char *train_images = malloc(TRAIN_IMAGES * IMAGE_SIZE);
    unsigned char *train_labels = malloc(TRAIN_IMAGES);
    unsigned char *test_images = malloc(TEST_IMAGES * IMAGE_SIZE);
    unsigned char *test_labels = malloc(TEST_IMAGES);

    if (!train_images || !train_labels || !test_images || !test_labels) {
        perror("Ошибка выделения памяти");
        exit(1);
    }

    // Загрузка данных
    printf("Чтение обучающих изображений...\n");
    read_images("train-images-idx3-ubyte", train_images, TRAIN_IMAGES);
    printf("Чтение обучающих меток...\n");
    read_labels("train-labels-idx1-ubyte", train_labels, TRAIN_IMAGES);
    printf("Чтение тестовых изображений...\n");
    read_images("t10k-images-idx3-ubyte", test_images, TEST_IMAGES);
    printf("Чтение тестовых меток...\n");
    read_labels("t10k-labels-idx1-ubyte", test_labels, TEST_IMAGES);

    // Инициализация весов
    double *w1 = malloc(INPUT_SIZE * HIDDEN_SIZE1 * sizeof(double));
    double *w2 = malloc(HIDDEN_SIZE1 * HIDDEN_SIZE2 * sizeof(double));
    double *w3 = malloc(HIDDEN_SIZE2 * OUTPUT_SIZE * sizeof(double));

    if (!w1 || !w2 || !w3) {
        perror("Ошибка выделения памяти для весов");
        exit(1);
    }

    initialize_weights(w1, INPUT_SIZE * HIDDEN_SIZE1, INPUT_SIZE);
    initialize_weights(w2, HIDDEN_SIZE1 * HIDDEN_SIZE2, HIDDEN_SIZE1);
    initialize_weights(w3, HIDDEN_SIZE2 * OUTPUT_SIZE, HIDDEN_SIZE2);

    // Обучение
    printf("Начало обучения...\n");
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf("Эпоха %d...\n", epoch + 1);
        for (int i = 0; i < TRAIN_IMAGES; i++) {
            double input[INPUT_SIZE];
            double target[OUTPUT_SIZE] = {0};

            // Нормализация входных данных
            for (int j = 0; j < INPUT_SIZE; j++) {
                input[j] = train_images[i * IMAGE_SIZE + j] / 255.0;
            }

            // One-hot encoding для целевого значения
            target[train_labels[i]] = 1.0;

            // Прямой проход и обратное распространение
            double hidden1[HIDDEN_SIZE1];
            double hidden2[HIDDEN_SIZE2];
            double output[OUTPUT_SIZE];
            forward_pass(input, hidden1, hidden2, output, w1, w2, w3);
            backpropagation(input, hidden1, hidden2, output, target, w1, w2, w3);
        }

        // Тестирование после каждой эпохи
        int correct = 0;
        for (int i = 0; i < TEST_IMAGES; i++) {
            double input[INPUT_SIZE];
            for (int j = 0; j < INPUT_SIZE; j++) {
                input[j] = test_images[i * IMAGE_SIZE + j] / 255.0;
            }

            double hidden1[HIDDEN_SIZE1];
            double hidden2[HIDDEN_SIZE2];
            double output[OUTPUT_SIZE];
            forward_pass(input, hidden1, hidden2, output, w1, w2, w3);

            // Нахождение максимального значения в выходном слое
            int predicted = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[predicted]) {
                    predicted = j;
                }
            }

            if (predicted == test_labels[i]) {
                correct++;
            }
        }

        // Вывод точности
        printf("Точность после эпохи %d: %.2f%%\n", epoch + 1, (double)correct / TEST_IMAGES * 100);
    }

    // Освобождение памяти
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);
    free(w1);
    free(w2);
    free(w3);

    return 0;
}