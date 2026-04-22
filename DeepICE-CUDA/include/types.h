#ifndef TYPES_H
#define TYPES_H

#define NUM_RELUS 2 // i.e. K
#define SIZE_ASGNS 64 // i.e. pow(2,K) *K

// use double precision data type
#define DOUBLE_PRECISION /* comment this to use single precision */
#ifdef DOUBLE_PRECISION
#define DATA_TYPE double
#else
#define DATA_TYPE float
#endif /* DOUBLE_PRCISION */

#define LABEL_TYPE int8_t

#define LOSS_TYPE int16_t


struct css_whole{
    int max_n;
    int max_r;
    unsigned* lens; // numbers of combinaitons so far for each r, it is C(n, r) for every r except max_r
    unsigned* max_lens;
    unsigned* whole_mem;
    unsigned** combs; // record each combinaitonss, which can be allocated first to achieve length==C(n,r), when n==N, r==D
};
#endif
