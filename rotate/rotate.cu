#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <cuda_runtime.h>

// Kernel CUDA para invertir los colores de la imagen
__global__ void invert_colors(unsigned char *d_image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 4; // Índice en el arreglo 1D (RGBA)
        d_image[idx] = 255 - d_image[idx];       // Invertir componente R
        d_image[idx + 1] = 255 - d_image[idx + 1]; // Invertir componente G
        d_image[idx + 2] = 255 - d_image[idx + 2]; // Invertir componente B
        // La componente A (alfa) se deja sin cambios
    }
}

// Función para leer un archivo PNG en el host
void read_png_file(const char *filename, int *width, int *height, unsigned char **image_data) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: No se pudo abrir el archivo %s para lectura.\n", filename);
        exit(EXIT_FAILURE);
    }

    unsigned char header[8];
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8)) {
        fprintf(stderr, "Error: %s no es un archivo PNG válido.\n", filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "Error: Fallo al inicializar la estructura de lectura de PNG.\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "Error: Fallo al inicializar la estructura de información de PNG.\n");
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error: Error durante la lectura de la imagen PNG.\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    *width = png_get_image_width(png_ptr, info_ptr);
    *height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    if (bit_depth == 16)
        png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    png_read_update_info(png_ptr, info_ptr);

    int row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    *image_data = (unsigned char*)malloc(row_bytes * (*height));

    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * (*height));
    for (int y = 0; y < *height; y++) {
        row_pointers[y] = (*image_data) + y * row_bytes;
    }

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);

    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
}

// Función para escribir un archivo PNG en el host
void write_png_file(const char *filename, int width, int height, unsigned char *image_data) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: No se pudo abrir el archivo %s para escritura.\n", filename);
        exit(EXIT_FAILURE);
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "Error: Fallo al inicializar la estructura de escritura de PNG.\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "Error: Fallo al inicializar la estructura de información de PNG.\n");
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error: Error durante la escritura de la imagen PNG.\n");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB_ALPHA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = image_data + y * width * 4; // 4 bytes por píxel (RGBA)
    }

    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);

    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main() {
    const char *input_filename = "example.png";
    const char *output_filename = "output.png";
    int width, height;
    unsigned char *image_data;

    // Leer la imagen PNG en el host
    read_png_file(input_filename, &width, &height, &image_data);

    // Asignar memoria en la GPU y copiar los datos de la imagen
    unsigned char *d_image;
    size_t image_size = width * height * 4 * sizeof(unsigned char);
    cudaMalloc((void**)&d_image, image_size);
    cudaMemcpy(d_image, image_data, image_size, cudaMemcpyHostToDevice);

    // Comprobar errores de CUDA
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Error en la copia de Host a Device: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Definir el tamaño del grid y de los bloques
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Ejecutar el kernel para invertir los colores
    invert_colors<<<gridDim, blockDim>>>(d_image, width, height);
    cudaDeviceSynchronize();

    // Comprobar errores de CUDA
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Error en el kernel: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Copiar los resultados de vuelta al host
    cudaMemcpy(image_data, d_image, image_size, cudaMemcpyDeviceToHost);

    // Comprobar errores de CUDA
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Error en la copia de Device a Host: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Escribir la imagen procesada a un nuevo archivo PNG
    write_png_file(output_filename, width, height, image_data);

    // Liberar memoria
    cudaFree(d_image);
    free(image_data);

    return 0;
}

/*

#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <cuda_runtime.h>

// Kernel CUDA para invertir los colores de la imagen
__global__ void invert_colors(unsigned char *d_image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // Índice en el arreglo 1D
        d_image[idx] = 255 - d_image[idx];       // Invertir componente R
        d_image[idx + 1] = 255 - d_image[idx + 1]; // Invertir componente G
        d_image[idx + 2] = 255 - d_image[idx + 2]; // Invertir componente B
    }
}

// Función para leer un archivo PNG en el host
void read_png_file(const char *filename, int *width, int *height, unsigned char **image_data) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: No se pudo abrir el archivo %s para lectura.\n", filename);
        exit(EXIT_FAILURE);
    }

    unsigned char header[8];
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8)) {
        fprintf(stderr, "Error: %s no es un archivo PNG válido.\n", filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "Error: Fallo al inicializar la estructura de lectura de PNG.\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "Error: Fallo al inicializar la estructura de información de PNG.\n");
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error: Error durante la lectura de la imagen PNG.\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    *width = png_get_image_width(png_ptr, info_ptr);
    *height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    printf("[%i,%i]\n",*width,*height);
    printf("unsigned chart=%lu, png_bytep=%lu\n",sizeof(unsigned char),sizeof(png_bytep));
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    png_read_update_info(png_ptr, info_ptr);

    int row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    *image_data = (unsigned char*)malloc(row_bytes * (*height));

    printf("rowbytes=%i,%i\n",row_bytes,(*width)*4);

    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * (*height));

    printf("image_data=%i,row_pointers=%lu\n",row_bytes * (*height), (sizeof(png_bytep) * (*height)*row_bytes));

    for (int y = 0; y < *height; y++) {
        row_pointers[y] = (*image_data) + y * row_bytes;
        //printf("%i\t%i\n",(row_pointers[y]),((*image_data) + y * row_bytes) );
        // row_pointers[y] = image_data[i]
    }
    
    png_read_image(png_ptr, row_pointers);

    //free(row_pointers);
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);

}

// Función para escribir un archivo PNG en el host
void write_png_file(const char *filename, int width, int height, unsigned char *image_data) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: No se pudo abrir el archivo %s para escritura.\n", filename);
        exit(EXIT_FAILURE);
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "Error: Fallo al inicializar la estructura de escritura de PNG.\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "Error: Fallo al inicializar la estructura de información de PNG.\n");
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error: Error durante la escritura de la imagen PNG.\n");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB_ALPHA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = image_data + y * width * 4; // 4 bytes por píxel (RGBA)
    }


    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);

    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main() {
    const char *input_filename = "example.png";
    const char *output_filename = "output.png";
    int width, height;
    unsigned char *image_data;

    // Leer la imagen PNG en el host
    read_png_file(input_filename, &width, &height, &image_data);

    // Asignar memoria en la GPU y copiar los datos de la imagen
    unsigned char *d_image;
    size_t image_size = width * height * 4 * sizeof(unsigned char);
    cudaMalloc((void**)&d_image, image_size);
    cudaMemcpy(d_image, image_data, image_size, cudaMemcpyHostToDevice);

 // Comprobar errores de CUDA
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "1:Error en el kernel: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }


    // Definir el tamaño del grid y de los bloques
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Ejecutar el kernel para invertir los colores
    invert_colors<<<gridDim, blockDim>>>(d_image, width, height);
    cudaDeviceSynchronize();

 // Comprobar errores de CUDA
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "2:Error en el kernel: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }


    // Copiar los resultados de vuelta al host
    cudaMemcpy(image_data, d_image, image_size, cudaMemcpyDeviceToHost);

 // Comprobar errores de CUDA
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "3:Error en el kernel: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }


    // Escribir la imagen procesada a un nuevo archivo PNG
    write_png_file(output_filename, width, height, image_data);

    // Liberar memoria
    cudaFree(d_image);
    free(image_data);

    return 0;
}

/*
#include <stdio.h>
#include <stdlib.h>
#include <png.h>

#define PNG_BYTES_TO_CHECK 8

void read_png_file(const char *filename, int *width, int *height, png_bytep **image_data, png_bytep **new_image_data) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: No se pudo abrir el archivo %s para lectura.\n", filename);
        exit(EXIT_FAILURE);
    }

    // Verificar la firma de la imagen PNG
    unsigned char header[PNG_BYTES_TO_CHECK];
    fread(header, 1, PNG_BYTES_TO_CHECK, fp);
    if (png_sig_cmp(header, 0, PNG_BYTES_TO_CHECK)) {
        fprintf(stderr, "Error: %s no es un archivo PNG válido.\n", filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Inicializar la estructura de lectura de PNG
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "Error: Fallo al inicializar la estructura de lectura de PNG.\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Inicializar la estructura de información de PNG
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "Error: Fallo al inicializar la estructura de información de PNG.\n");
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Configurar manejo de errores de libpng
    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error: Error durante la lectura de la imagen PNG.\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Configurar libpng para leer desde el archivo
    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, PNG_BYTES_TO_CHECK);

    // Leer la información de la imagen PNG
    png_read_info(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);

    // Convertir a formato RGB de 24 bits si es necesario
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    // Actualizar la información de la imagen
    png_read_update_info(png_ptr, info_ptr);

    // Obtener la información de la imagen
    *width = png_get_image_width(png_ptr, info_ptr);
    *height = png_get_image_height(png_ptr, info_ptr);
    int row_bytes = png_get_rowbytes(png_ptr, info_ptr);

    // Reservar memoria para los datos de la imagen
    *image_data = (png_bytep*)malloc(sizeof(png_bytep) * (*height));
    for (int i = 0; i < *height; i++) {
        (*image_data)[i] = (png_byte*)malloc(row_bytes);
    }

    // Reservar memoria para los datos de la imagen
    *new_image_data = (png_bytep*)malloc(sizeof(png_bytep) * (*height));
    for (int i = 0; i < *height; i++) {
        (*new_image_data)[i] = (png_byte*)malloc(row_bytes);
    }


    // Leer los datos de la imagen
    png_bytep row_pointers[*height];
    for (int i = 0; i < *height; i++) {
        row_pointers[i] = (*image_data)[i];
    }
    png_read_image(png_ptr, row_pointers);

    // Finalizar la lectura
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
}

void free_image_data(png_bytep *image_data, int height) {
    for (int i = 0; i < height; i++) {
        free(image_data[i]);
    }
    free(image_data);
}

__global__ void rotatePoints(const float *x, const float *y, float *x_new, float *y_new, float cosTheta, float sinTheta, int numPoints) {
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < numPoints) {
x_new[i] = x[i] * cosTheta - y[i] * sinTheta;
y_new[i] = x[i] * sinTheta + y[i] * cosTheta;
}
}

int main() {
    const char *filename = "example.png";
    int width, height;
    png_bytep *image_data;
    png_bytep *new_image_data;

    read_png_file(filename, &width, &height, &image_data, &new_image_data);

    // Imprimir información de la imagen
    printf("Ancho: %d, Alto: %d\n", width, height);


    // Ángulo de rotación (por ejemplo, 45 grados)
    float theta = M_PI / 4;
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    // Reservar memoria en el dispositivo

    float *d_x = NULL;
    cudaMalloc((void **)&d_x, height);

    *image_data = (png_bytep*)malloc(sizeof(png_bytep) * (*height));
    for (int i = 0; i < *height; i++) {
        (*image_data)[i] = (png_byte*)malloc(row_bytes);
    }



    float *d_y = NULL;
    cudaMalloc((void **)&d_y, height);


    float *d_x_new = NULL;
    cudaMalloc((void **)&d_x_new, width);
    float *d_y_new = NULL;
    cudaMalloc((void **)&d_y_new, height);

    // Copiar datos desde el host al dispositivo
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);


       // Acceder a los datos de la imagen
    for (int y = 0; y < height; y++) {
        png_bytep row = image_data[y];



        for (int x = 0; x < width * 3; x += 3) { // Cada píxel RGB ocupa 3 bytes
            // Acceder a los valores R, G, B del píxel
            png_byte r = row[x];
            png_byte g = row[x + 1];
            png_byte b = row[x + 2];
            // Hacer algo con los valores (por ejemplo, imprimirlos)
            printf("Pixel en (%d, %d): R=%d, G=%d, B=%d\n", x / 3, y, r, g, b);
        }
    }

    // Liberar memoria después de su uso
    free_image_data(image_data, height);

    return 0;
}
*/