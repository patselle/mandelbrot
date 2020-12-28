#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>
#include "pyfrac.h"

void pyfrac_zoom(
	char * const r_min_str,
	char * const r_max_str,
	char * const i_min_str,
	char * const i_max_str,
	unsigned int const str_len,
	unsigned int const precision,
    char const * const r_str,
    char const * const i_str,
    float const percent)
{
	mpfr_t r_min, i_min, r_max, i_max, r_point, i_point, new_r_min, new_i_min, new_r_max, new_i_max, r_dim, i_dim, new_r_dim, new_i_dim, tmp, r_rel, i_rel, r_rel2, i_rel2, r_off, i_off;
	
	mpfr_inits2(precision, r_min, i_min, r_max, i_max, r_point, i_point, new_r_min, new_i_min, new_r_max, new_i_max, r_dim, i_dim, new_r_dim, new_i_dim, tmp, r_rel, i_rel, r_rel2, i_rel2, r_off, i_off, (mpfr_ptr) 0);
	
	mpfr_set_str(r_min, r_min_str, 10, MPFR_RNDN);
	mpfr_set_str(i_min, i_min_str, 10, MPFR_RNDN);
	mpfr_set_str(r_max, r_max_str, 10, MPFR_RNDN);
	mpfr_set_str(i_max, i_max_str, 10, MPFR_RNDN);
	mpfr_set_str(r_point, r_str, 10, MPFR_RNDN);
	mpfr_set_str(i_point, i_str, 10, MPFR_RNDN);

	// dim = max - min
	mpfr_sub(r_dim, r_max, r_min, MPFR_RNDN);
	mpfr_sub(i_dim, i_max, i_min, MPFR_RNDN);

	// ndim = dim * %
	mpfr_mul_d(new_r_dim, r_dim, 1 - percent, MPFR_RNDN);
	mpfr_mul_d(new_i_dim, i_dim, 1 - percent, MPFR_RNDN);

	// off = (dim - ndim) / 2
	mpfr_sub(tmp, r_dim, new_r_dim, MPFR_RNDN);
	mpfr_div_ui(r_off, tmp, 2, MPFR_RNDN);
	
	mpfr_sub(tmp, i_dim, new_i_dim, MPFR_RNDN);
	mpfr_div_ui(i_off, tmp, 2, MPFR_RNDN);
	
	// rel = 2 * (P - min) / dim;
	mpfr_sub(r_rel, r_point, r_min, MPFR_RNDN); 
	mpfr_mul_ui(tmp, r_rel, 2, MPFR_RNDN);
	mpfr_div(r_rel, tmp, r_dim, MPFR_RNDN);	
	
	mpfr_sub(i_rel, i_point, i_min, MPFR_RNDN); 
	mpfr_mul_ui(tmp, i_rel, 2, MPFR_RNDN);
	mpfr_div(i_rel, tmp, i_dim, MPFR_RNDN);	
	
	// rel2 = rel^2
	mpfr_mul(r_rel2, r_rel, r_rel, MPFR_RNDN);
	mpfr_mul(i_rel2, i_rel, i_rel, MPFR_RNDN);
	
	// nmin = rel2 * off + min
	mpfr_mul(tmp, r_rel2, r_off, MPFR_RNDN);
	mpfr_add(new_r_min, tmp, r_min, MPFR_RNDN);
	
	mpfr_mul(tmp, i_rel2, i_off, MPFR_RNDN);
	mpfr_add(new_i_min, tmp, i_min, MPFR_RNDN);
	
	// nmax = nmin + ndim
	mpfr_add(new_r_max, new_r_min, new_r_dim, MPFR_RNDN);
	mpfr_add(new_i_max, new_i_min, new_i_dim, MPFR_RNDN);

	// finish
    pyfrac_str(r_min_str, str_len, new_r_min);
    pyfrac_str(r_max_str, str_len, new_r_max);
    pyfrac_str(i_min_str, str_len, new_i_min);
    pyfrac_str(i_max_str, str_len, new_i_max);

	mpfr_clears(i_min, r_max, i_max, r_point, i_point, new_r_min, new_i_min, new_r_max, new_i_max, r_dim, i_dim, new_r_dim, new_i_dim, tmp, r_rel, i_rel, r_rel2, i_rel2, r_off, i_off, (mpfr_ptr) 0);
	mpfr_free_cache();
}