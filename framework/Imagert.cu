
#include "Imager.h"


void gamma_correct_host(float *src, float *dst, int w, int h, int nc, float g)
{
    for (int i = 0; i < w*h*nc; ++i)
    {
        dst[i] = pow(src[i], g);
    }
}
