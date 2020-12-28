#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>
#include "pyfrac.h"

#define ABS(n) ((n >> 31) ^ n) - (n >> 31)

void pyfrac_str(char * const str, const unsigned int size, mpfr_t op)
{
	char *buf;
	buf = malloc(sizeof(char) * (size + 1));

	mpfr_exp_t e;
	mpfr_get_str(buf, &e, 10, size, op, MPFR_RNDN);

	memset(str, '0', size);


	if (e == 0)
	{
		if (buf[0] == '-')
		{
			memcpy(str + 3, buf + 1, size - 3);
			str[0] = '-';
			str[2] = '.';
		}
		else
		{
			memcpy(str + 2, buf, size - 2);
			str[1] = '.';
		}
	}
	else if (e > 0)
	{
		if (buf[0] == '-')
		{
			memcpy(str, buf, e + 1);
			memcpy(str + e + 2, buf + e + 1, size - e - 2);
			str[e + 1] = '.';
		}
		else
		{
			memcpy(str, buf, e);
			memcpy(str + e + 1, buf + e, size - e - 1);
			str[e] = '.';
		}
	}
	else if (ABS(e) < size)
	{
		if (buf[0] == '-')
		{
			memcpy(str - e + 3, buf + 1, size + e - 3);
			str[0] = '-';
			str[2] = '.';
		}
		else
		{
			memcpy(str - e + 2, buf, size + e - 2);
			str[1] = '.';
		}
	}

	free(buf);
}