#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>
#include "pyfrac.h"

void pyfrac_frame(
    unsigned int * const frame,
    unsigned int const width,
    unsigned int const height,
    unsigned int const row_begin,
    unsigned int const row_height,
    char const * const r_min_str,
    char const * const r_max_str,
    char const * const i_min_str,
    char const * const i_max_str,
    unsigned int const precision,
    unsigned int const threshold,
    unsigned int const limit)
{
	unsigned int x, y, i;

	mpfr_t r_min, r_max, i_min, i_max, r_step, i_step, tmp, c_r, c_i, z_r, z_i, z_r2, z_i2;	
	mpfr_inits2(precision, r_min, r_max, i_min, i_max, r_step, i_step, tmp, c_r, c_i, z_r, z_i, z_r2, z_i2, (mpfr_ptr) 0);

	mpfr_set_str(r_min, r_min_str, 10, MPFR_RNDN);
	mpfr_set_str(r_max, r_max_str, 10, MPFR_RNDN);
	mpfr_set_str(i_min, i_min_str, 10, MPFR_RNDN);
	mpfr_set_str(i_max, i_max_str, 10, MPFR_RNDN);

	// compute step size

	mpfr_sub(r_step, r_max, r_min, MPFR_RNDN);
	mpfr_div_ui(r_step, r_step, width, MPFR_RNDN);
	mpfr_sub(i_step, i_max, i_min, MPFR_RNDN);
	mpfr_div_ui(i_step, i_step, height, MPFR_RNDN);

	// compute partial frame
	for (y = 0; y < row_height; y++)
	{
		for (x = 0; x < width; x++)
		{
			i = 0;
			// init c.r
			mpfr_mul_ui(tmp, r_step, x, MPFR_RNDN);
			mpfr_add(c_r, r_min, tmp, MPFR_RNDN);

			// init c.i
			mpfr_mul_ui(tmp, i_step, row_begin + y, MPFR_RNDN);
			mpfr_add(c_i, i_min, tmp, MPFR_RNDN);

			// init z
			mpfr_set_ui(z_r, 0, MPFR_RNDN);
			mpfr_set_ui(z_i, 0, MPFR_RNDN);
			while (1)
			{
				// compute z squared
				//   compute z_r squared
				//   z_r^2 = z_r^2 - z_i^2
				mpfr_mul(z_r2, z_r, z_r, MPFR_RNDN);
				mpfr_mul(tmp, z_i, z_i, MPFR_RNDN);
				mpfr_sub(z_r2, z_r2, tmp, MPFR_RNDN);

				//   compute z_i squared
				//   z_i^2 = 2 * z_r * z_i
				mpfr_mul(z_i2, z_r, z_i, MPFR_RNDN);
				mpfr_mul_ui(z_i2, z_i2, 2, MPFR_RNDN);

				// compute absolute value of z squared
				// |z^2| = z_r^2 + z_i^2
				mpfr_add(tmp, z_r2, z_i2, MPFR_RNDN);
				// compare absolute value of z squared with threshold
				if (mpfr_cmp_ui(tmp, threshold) >= 0)
				{
					frame[y * width + x] = i;
					break;
				}

				i++;

				if (i == limit)
				{
					frame[y * width + x] = i - 1;
					break;
				}

				// assign z = z^2 + c
				mpfr_add(z_r, z_r2, c_r, MPFR_RNDN);
				mpfr_add(z_i, z_i2, c_i, MPFR_RNDN);
			}
		}
	}

	// clean up

	mpfr_clears(r_min, r_max, i_min, i_max, r_step, i_step, tmp, c_r, c_i, z_r, z_i, z_r2, z_i2, (mpfr_ptr) 0);
	mpfr_free_cache();
}
