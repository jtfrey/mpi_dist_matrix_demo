

#ifndef __ME_KERNEL_H__
#define __ME_KERNEL_H__

#include "int_pair.h"

static const char *me_kernel_description = "A_{i,j} = Sqrt[i*i + j*j]";

static inline double
me_kernel(
    int_pair_t  p
)
{
    double      i = (double)p.i,
                j = (double)p.j;
    return sqrt(i * i + j * j);
}

#endif /* __ME_KERNEL_H__ */
