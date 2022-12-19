#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sobel.h"
#include "main_testbench.h"

uint8_t output[HEIGHT*WIDTH];

int main(int argc, char *argv[]) {
    int errors = 0;

	sobel(output, input, HEIGHT, WIDTH);

    // Check errors
    for(int i = 0; i < HEIGHT*WIDTH; i++)
        if(output[i] != g_output[i])
           errors++;

    printf("Correctness: %.2f %% (%d Errors)\n", (1.0 - ((float) errors / (float)(HEIGHT*WIDTH)))*100.0, errors);

    return errors;
}
