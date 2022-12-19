#include <math.h>
#include "sobel.h"
#include <cstring>
#include <cstdio>

#define HEIGHT 512
#define WIDTH 512

void sobel(uint8_t *__restrict__ out, uint8_t *__restrict__ in, int width, int height)
{
    #pragma HLS INTERFACE m_axi port=in offset=slave bundle=in_mem
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=out_mem
    #pragma HLS INTERFACE s_axilite port=width  bundle=param
    #pragma HLS INTERFACE s_axilite port=height bundle=param
    #pragma HLS INTERFACE s_axilite port=return bundle=param

    int sobelFilter[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

    #pragma HLS ARRAY_PARTITION variable=sobelFilter dim=0 complete

    uint8_t pezzo[4*WIDTH];

    #pragma HLS ARRAY_PARTITION variable=pezzo dim=0 factor=3 block

    memcpy(pezzo, in, sizeof(uint8_t)*WIDTH*3);
    for (int y = 1; y < height - 1; y++)
    {

        #pragma HLS LOOP_TRIPCOUNT max=HEIGHT min=HEIGHT

        for (int x = 1; x < width - 1; x++)
        {
            #pragma HLS LOOP_TRIPCOUNT max=WIDTH min=WIDTH

            int dx = 0, dy = 0;

            loop_3: for (int k = 0; k < 3; k++)
            {
                int yi = ((y + k - 1) % 4) * WIDTH;
                #pragma HLS UNROLL
                #pragma HLS LOOP_TRIPCOUNT max=3 min=3
               loop_4:  for (int z = 0; z < 3; z++)
                {
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT max=3 min=3

                    uint8_t v = pezzo[yi + x + z - 1];
                    dx += sobelFilter[k][z] * v;
                    dy += sobelFilter[z][k] * v;
                }
            }
            out[y * width + x] = sqrt((float)((dx * dx) + (dy * dy)));
        }
        //printf("MEMCPY: %i %i\n", ((y + 2) % 3) * WIDTH, (y + 2)*WIDTH);
        memcpy(&pezzo[((y + 2) % 4) * WIDTH], &in[(y + 2)*WIDTH], sizeof(uint8_t)*WIDTH);
    }
}
