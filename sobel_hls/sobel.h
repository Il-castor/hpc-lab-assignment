#ifndef SOBEL_H
#define SOBEL_H

#include <stdint.h>

void sobel(uint8_t *__restrict__ out, uint8_t *__restrict__ in, int width, int height);

#endif
