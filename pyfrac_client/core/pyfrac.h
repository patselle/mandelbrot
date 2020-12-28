#ifndef __PYFRAC_H
#define __PYFRAC_H

#include <mpfr.h>

#define RGB2INT(r,g,b) ((unsigned int)r << 16) | ((unsigned int)g << 8) | ((unsigned int)b << 0)
#define INT2R(i) ((i >> 16) & 0xff)
#define INT2G(i) ((i >> 8) & 0xff) 
#define INT2B(i) ((i >> 0) & 0xff) 

enum
{
    BLACK = 0,
    WHITE = RGB2INT(255, 255, 255),
    RED = RGB2INT(255, 0, 0),
    GREEN = RGB2INT(0, 255, 0),
    BLUE = RGB2INT(0, 0, 255),
};

typedef struct
{
    unsigned int width;
    unsigned int height;
    char *r_min;
	char *i_min;
	char *r_max;
	char *i_max;
    char *r_step;
    char *i_step;
    unsigned int precision;
    unsigned int str_len;
    unsigned int threshold;
    unsigned int limit;
    unsigned int *frame;
    unsigned int *map;
} pyfrac_config_t;

typedef struct
{
    unsigned int *data;
    unsigned int data_len;
    unsigned int padding;
    unsigned int *biases;
    unsigned int biases_len;
} pyfrac_palette_t;

// utils

extern void pyfrac_str(char * const, const unsigned int, mpfr_t);

// frame

extern void pyfrac_frame(
    unsigned int * const frame,
    unsigned int const width,
    unsigned int const height,
    unsigned int const row_begin,
    unsigned int const row_height,
    char const * const r_min,
    char const * const r_max,
    char const * const i_min,
    char const * const i_max,
    unsigned int const precision,
    unsigned int const threshold,
    unsigned int const limit);

// create bitmap

extern void pyfrac_bmp(
    char const * const path,
    unsigned int const width,
    unsigned int const height,
    unsigned int const * const frame);

// zoom

extern void pyfrac_zoom(
	char * const r_min_str,
	char * const r_max_str,
	char * const i_min_str,
	char * const i_max_str,
	unsigned int const str_len,
    unsigned int const precision,
    char const * const r_str,
    char const * const i_str,
    float const percent);

#endif