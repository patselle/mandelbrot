#include <stdio.h>
#include <stdlib.h>
#include "pyfrac.h"

#define HEADER_SIZE 54

static unsigned char bmppad[3] = { 0, 0, 0 };

static void pyfrac_bmp_header(FILE * const f, unsigned int const width, unsigned int const height)
{
    unsigned int file_size = HEADER_SIZE + 3 * width * height;

    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};

    bmpfileheader[ 2] = (unsigned char)(file_size      );
    bmpfileheader[ 3] = (unsigned char)(file_size >>  8);
    bmpfileheader[ 4] = (unsigned char)(file_size >> 16);
    bmpfileheader[ 5] = (unsigned char)(file_size >> 24);

    bmpinfoheader[ 4] = (unsigned char)(width      );
    bmpinfoheader[ 5] = (unsigned char)(width >>  8);
    bmpinfoheader[ 6] = (unsigned char)(width >> 16);
    bmpinfoheader[ 7] = (unsigned char)(width >> 24);
    bmpinfoheader[ 8] = (unsigned char)(height      );
    bmpinfoheader[ 9] = (unsigned char)(height >>  8);
    bmpinfoheader[10] = (unsigned char)(height >> 16);
    bmpinfoheader[11] = (unsigned char)(height >> 24);

    fwrite(bmpfileheader, 1, 14, f);
    fwrite(bmpinfoheader, 1, 40, f);
}

void pyfrac_bmp(char const * const path, unsigned int const width, unsigned int const height, unsigned int const * const frame)
{
    unsigned int i, j, x, y;
    unsigned char *img;
    FILE *f;

    img = (unsigned char*)malloc(3 * width * height);
    memset(img, 0, 3 * width * height);

    for(x = 0; x < width; x++)
    {
        for(y = 0; y < height; y++)
        {
            i = (height - 1) - y;
            j = frame[y * width + x];

            img[(i * width + x) * 3 + 2] = (unsigned char)(INT2R(j));
            img[(i * width + x) * 3 + 1] = (unsigned char)(INT2G(j));
            img[(i * width + x) * 3 + 0] = (unsigned char)(INT2B(j));
        }
    }

    f = fopen(path,"wb");

    pyfrac_bmp_header(f, width, height);

    for(i = 0; i < height; i++)
    {
        fwrite(img + (width * (height - i - 1) * 3), 3, width, f);
        fwrite(bmppad, 1, (4 - (width * 3) % 4) % 4, f);
    }

    free(img);
    fclose(f);
}