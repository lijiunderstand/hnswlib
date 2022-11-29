#pragma once
#include "hnswlib.h"

namespace hnswlib {

    static float
    L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float t = *pVect1 - *pVect2;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return (res);
    }

#if defined(USE_AVX512)

    // Favor using AVX512 if available.
    static float
    L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN64 TmpRes[16];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m512 diff, v1, v2;
        __m512 sum = _mm512_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm512_loadu_ps(pVect1);
            pVect1 += 16;
            v2 = _mm512_loadu_ps(pVect2);
            pVect2 += 16;
            diff = _mm512_sub_ps(v1, v2);
            // sum = _mm512_fmadd_ps(diff, diff, sum);
            sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
        }

        _mm512_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

        return (res);
}
#endif

#if defined(USE_AVX)

    // Favor using AVX if available.
    static float
    L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }

#endif

#if defined(USE_SSE)

    static float
    L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    DISTFUNC<float> L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;

    static float
    L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
        float *pVect1 = (float *) pVect1v + qty16;
        float *pVect2 = (float *) pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
        return (res + res_tail);
    }
#endif


#if defined(USE_SSE)
    static float
    L2SqrSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);


        size_t qty4 = qty >> 2;

        const float *pEnd1 = pVect1 + (qty4 << 2);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }
#endif

#if defined(USE_AVX)
    static float
    L2SqrSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);


        // size_t qty4 = qty >> 2;
        size_t qty16 = qty / 16;
        size_t qty4 = qty / 4;

        // const float *pEnd1 = pVect1 + (qty4 << 2);

        const float *pEnd1 = pVect1 + 16 * qty16;
        const float *pEnd2 = pVect1 + 4 * qty4;

        __m256 sum256 = _mm256_set1_ps(0);


        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m256 v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            __m256 v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            __m256 diff = _mm256_sub_ps(v1, v2);
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff =  _mm256_sub_ps(v1, v2);
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(diff, diff));
        }
        __m128 v1, v2, diff;
        __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

        while (pVect1 < pEnd2) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(diff, diff));
        }

        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];;
        return sum;
    }

#endif

#if defined(USE_AVX512)

 static float
    L2SqrSIMD4ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);


        // size_t qty4 = qty >> 2;
        size_t qty16 = qty / 16;
        size_t qty4 = qty / 4;

        // const float *pEnd1 = pVect1 + (qty4 << 2);

        const float *pEnd1 = pVect1 + 16 * qty16;
        const float *pEnd2 = pVect1 + 4 * qty4;

        __m512 sum512 = _mm512_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m512 v1 = _mm512_loadu_ps(pVect1);
            pVect1 += 16;
            __m512 v2 = _mm512_loadu_ps(pVect2);
            pVect2 += 16;
            __m512 diff = _mm512_sub_ps(v1, v2);
            sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(diff, diff));
        }

        __m256 v1, v2, diff;
        __m256 sum_prod = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));

        while (pVect1 < pEnd2) {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm256_sub_ps(v1, v2);
            sum_prod = _mm256_add_ps(sum_prod, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];;
        return sum;

    }

#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512) 
    DISTFUNC<float> L2SqrSIMD4Ext = L2SqrSIMD4ExtSSE;
    static float
    L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
        size_t qty_left = qty - qty4;

        float *pVect1 = (float *) pVect1v + qty4;
        float *pVect2 = (float *) pVect2v + qty4;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

        return (res + res_tail);
    }
#endif

    class L2Space : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2Space(size_t dim) {
            fstdistfunc_ = L2Sqr;
    #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
        #if defined(USE_AVX512)
            if (AVX512Capable()){
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
                L2SqrSIMD4Ext  =  L2SqrSIMD4ExtAVX512;}
            else if (AVXCapable()){
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
                L2SqrSIMD4Ext  =  L2SqrSIMD4ExtAVX;}
        #elif defined(USE_AVX)
            if (AVXCapable()){
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
                L2SqrSIMD4Ext  =  L2SqrSIMD4ExtAVX;}
        #endif

            if (dim % 16 == 0)
                fstdistfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = L2SqrSIMD4ExtResiduals;
    #endif
            dim_ = dim;
            data_size_ = dim * sizeof(float);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2Space() {}
    };

    static int
    L2SqrI16x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {

        size_t qty = *((size_t *) qty_ptr);
        int res = 0;
        unsigned char *a = (unsigned char *) pVect1;
        unsigned char *b = (unsigned char *) pVect2;

        qty = qty >> 4;// dimension/4
        for (size_t i = 0; i < qty; i++) {

            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return (res);
    }
    static int
    L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {

        size_t qty = *((size_t *) qty_ptr);
        int res = 0;
        unsigned char *a = (unsigned char *) pVect1;
        unsigned char *b = (unsigned char *) pVect2;

        qty = qty >> 2;// dimension/4
        for (size_t i = 0; i < qty; i++) {

            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return (res);
    }
    static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
        size_t qty = *((size_t*)qty_ptr);
        int res = 0;
        unsigned char* a = (unsigned char*)pVect1;
        unsigned char* b = (unsigned char*)pVect2;

        for(size_t i = 0; i < qty; i++)
        {
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return (res);
    }
#if defined(USE_AVX512)

    // Favor using AVX512 if available.
    static int
    L2SqrI16xAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        // float *pVect1 = (float *) pVect1v;
        // float *pVect2 = (float *) pVect2v;
        // size_t qty = *((size_t *) qty_ptr);
        size_t qty = *((size_t*)qty_ptr);
        int res = 0;
        // unsigned char* pVect1 = (unsigned char*)pVect1;
        // unsigned char* pVect2= (unsigned char*)pVect2;

        const __m512i* pVect1 = reinterpret_cast<const __m512i*>(pVect1);
        const __m512i* pVect2 = reinterpret_cast<const __m512i*>(pVect2);
  

        int PORTABLE_ALIGN64 TmpRes[16];
        size_t qty16 = qty >> 4;
        const __m512i* pEnd1 = reinterpret_cast<const __m512i*>(pVect1+16*qty16);
        // const int *pEnd1 = pVect1 + (qty16 << 4);

        __m512i diff, v1, v2;
        __m512i sum = _mm512_setzero_si512();

        while (pVect1 < pEnd1) {
            v1 = _mm512_loadu_si512(pVect1);
            pVect1 += 16;
            v2 = _mm512_loadu_si512(pVect2);
            pVect2 += 16;
            diff = _mm512_sub_epi32(v1, v2);
            // sum = _mm512_fmadd_ps(diff, diff, sum);
            sum = _mm512_add_epi32(sum, _mm512_mul_epi32(diff, diff));
        }

        _mm512_store_epi32(TmpRes, sum);
        res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

        return (res);
    }

    // static int
    // L2SqrI4xAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {

    // size_t qty = *((size_t*)qty_ptr);
    // int res = 0;


    // const __m512i* pVect1 = reinterpret_cast<const __m512i*>(pVect1);
    // const __m512i* pVect2 = reinterpret_cast<const __m512i*>(pVect2);
  

    // int PORTABLE_ALIGN64 TmpRes[16];
    // size_t qty16 = qty >> 4;
    // size_t qty4 = qty >> 2;
    // const __m512i* pEnd1 = reinterpret_cast<const __m512i*>(pVect1+16*qty16);
    // const __m512i* pEnd2 = reinterpret_cast<const __m512i*>(pVect1+4*qty4);


    // // __m512i diff, v1, v2;
    // __m512i sum512 = _mm512_setzero_si512();

    // while (pVect1 < pEnd1) {
    //     __m512i v1 = _mm512_loadu_si512(pVect1);
    //     pVect1 += 16;
    //     __m512i v2 = _mm512_loadu_si512(pVect2);
    //     pVect2 += 16;
    //     __m512i diff = _mm512_sub_epi32(v1, v2);
    //     sum512 = _mm512_add_epi32(sum512, _mm512_mul_epi32(diff, diff));
    // }

    // __m256i v1, v2, diff;
    // __m256i sum_prod = _mm256_add_epi32(_mm512_extracti32x8_epi32(sum512, 0), _mm512_extracti32x8_epi32(sum512, 1));

    // while (pVect1 < pEnd2) {
    //     v1 = _mm256_loadu_si256(pVect1);
    //     pVect1 += 4;
    //     v2 = _mm256_loadu_si256(pVect2);
    //     pVect2 += 4;
    //     diff = _mm256_sub_epi32(v1, v2);
    //     sum_prod = _mm256_add_epi32(sum_prod, _mm256_mul_epi32(diff, diff));
    // }

    // _mm256_store_epi32((__m256i*)TmpRes, sum_prod);
    // res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];;
    // return res;    

    // }
#endif

#if defined(USE_AVX)
    static int
    L2SqrI16xAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        int PORTABLE_ALIGN32 TmpRes[8];

        size_t qty = *((size_t*)qty_ptr);
        int res = 0;
        const __m256i* pVect1 = reinterpret_cast<const __m256i*>(pVect1);
        const __m256i* pVect2 = reinterpret_cast<const __m256i*>(pVect2);
  
        size_t qty16 = qty >> 4;
        // size_t qty4 = qty >> 2;
        const __m256i* pEnd1 = reinterpret_cast<const __m256i*>(pVect1+16*qty16);
        // const __m256i* pEnd2 = reinterpret_cast<const __m256i*>(pVect1+4*qty4);


        // __m256 sum256 = _mm256_set1_epi32(0);

        
        __m256i diff, v1, v2;
        __m256i sum = _mm256_set1_epi32(0);

        while (pVect1 < pEnd1) {
            v1 = _mm256_loadu_si256(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_si256(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_epi32(v1, v2);
            sum = _mm256_add_epi32(sum, _mm256_mul_epi32(diff, diff));

            v1 = _mm256_loadu_si256(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_si256(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_epi32(v1, v2);
            sum = _mm256_add_epi32(sum, _mm256_mul_epi32(diff, diff));
        }
        // _mm256_store_si256(TmpRes, sum);  //you wen ti
        _mm256_store_si256((__m256i*)TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];



        // while (pVect1 < pEnd1) {
        //     //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        //     __m256 v1 = _mm256_loadu_epi32(pVect1);
        //     pVect1 += 8;
        //     __m256 v2 = _mm256_loadu_epi32(pVect2);
        //     pVect2 += 8;
        //     __m256 diff = _mm256_sub_epi32(v1, v2);
        //     sum256 = _mm256_add_epi32(sum256, _mm256_mul_epi32(diff, diff));

        //     v1 = _mm256_loadu_epi32(pVect1);
        //     pVect1 += 8;
        //     v2 = _mm256_loadu_epi32(pVect2);
        //     pVect2 += 8;
        //     diff =  _mm256_sub_epi32(v1, v2);
        //     sum256 = _mm256_add_epi32(sum256, _mm256_mul_epi32(diff, diff));
        // }
        // __m128 v1, v2, diff;
        // __m128 sum_prod = _mm_add_epi32(_mm256_extract_epi32(sum256, 0), _mm256_extract_epi32(sum256, 1));

        // while (pVect1 < pEnd2) {
        //     v1 = _mm_loadu_epi32(pVect1);
        //     pVect1 += 4;
        //     v2 = _mm_loadu_epi32(pVect2);
        //     pVect2 += 4;
        //     diff = _mm_sub_epi32(v1, v2);
        //     sum_prod = _mm_add_epi32(sum_prod, _mm_mul_epi32(diff, diff));
        // }

        // _mm_store_epi32(TmpRes, sum_prod);
        // float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];;
        // return sum;

    }


    // static int
    // L2SqrI4xAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    // int PORTABLE_ALIGN32 TmpRes[8];

    // size_t qty = *((size_t*)qty_ptr);
    // int res = 0;
    // const __m256i* pVect1 = reinterpret_cast<const __m256i*>(pVect1);
    // const __m256i* pVect2 = reinterpret_cast<const __m256i*>(pVect2);

    // size_t qty16 = qty >> 4;
    // size_t qty4 = qty >> 2;
    // const __m256i* pEnd1 = reinterpret_cast<const __m256i*>(pVect1+16*qty16);
    // const __m256i* pEnd2 = reinterpret_cast<const __m256i*>(pVect1+4*qty4);


    // // __m256 sum256 = _mm256_set1_epi32(0);

    
    // // __m256i diff, v1, v2;
    // __m256i sum256 = _mm256_set1_epi32(0);


    // while (pVect1 < pEnd1) {
    //     //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

    //     __m256i v1 = _mm256_loadu_si256(pVect1);
    //     pVect1 += 8;
    //     __m256i v2 = _mm256_loadu_si256(pVect2);
    //     pVect2 += 8;
    //     __m256i diff = _mm256_sub_epi32(v1, v2);
    //     sum256 = _mm256_add_epi32(sum256, _mm256_mul_epi32(diff, diff));

    //     v1 = _mm256_loadu_si256(pVect1);
    //     pVect1 += 8;
    //     v2 = _mm256_loadu_si256(pVect2);
    //     pVect2 += 8;
    //     diff =  _mm256_sub_epi32(v1, v2);
    //     sum256 = _mm256_add_epi32(sum256, _mm256_mul_epi32(diff, diff));
    // }
    // __m128i v1, v2, diff;
    // __m128i sum_prod = _mm_add_epi32(_mm256_extract_epi32(sum256, 0), _mm256_extract_epi32(sum256, 1));

    // while (pVect1 < pEnd2) {
    //     v1 = _mm_loadu_si128(pVect1);
    //     pVect1 += 4;
    //     v2 = _mm_loadu_si128(pVect2);
    //     pVect2 += 4;
    //     diff = _mm_sub_epi32(v1, v2);
    //     sum_prod = _mm_add_epi32(sum_prod, _mm_mul_epi32(diff, diff));
    // }

    // _mm_store_si128((__m128i*)TmpRes, sum_prod);
    // res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];;
    // return res;

    // }

#endif

#if defined(USE_SSE)
    static int
    L2SqrI16xSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {

        int PORTABLE_ALIGN32 TmpRes[8];
        size_t qty = *((size_t*)qty_ptr);
        int res = 0;
        const __m128i* pVect1 = reinterpret_cast<const __m128i*>(pVect1);
        const __m128i* pVect2 = reinterpret_cast<const __m128i*>(pVect2);
        size_t qty16 = qty >> 4;
        const __m128i* pEnd1 = reinterpret_cast<const __m128i*>(pVect1+16*qty16);

        // unsigned char *pVect1 = (unsigned char *) pVect1;
        // unsigned char *pVect2 = (unsigned char *) pVect2;
        // unsigned char *pEnd1 = pVect1 +16*qty16;

        __m128i diff, v1, v2;
        __m128i sum = _mm_set1_epi32(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
            v1 = _mm_loadu_si128(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_si128 (pVect2);
            pVect2 += 4;
            diff = _mm_sub_epi32(v1, v2);
            sum = _mm_add_epi32(sum, _mm_mul_epi32(diff, diff));

            v1 = _mm_loadu_si128(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_si128(pVect2);
            pVect2 += 4;
            diff = _mm_sub_epi32(v1, v2);
            sum = _mm_add_epi32(sum, _mm_mul_epi32(diff, diff));

            v1 = _mm_loadu_si128(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_si128(pVect2);
            pVect2 += 4;
            diff = _mm_sub_epi32(v1, v2);
            sum = _mm_add_epi32(sum, _mm_mul_epi32(diff, diff));

            v1 = _mm_loadu_si128(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_si128(pVect2);
            pVect2 += 4;
            diff = _mm_sub_epi32(v1, v2);
            sum = _mm_add_epi32(sum, _mm_mul_epi32(diff, diff));
        }

        _mm_store_si128((__m128i*)TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    }

    static int
    L2SqrI4xSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        int PORTABLE_ALIGN32 TmpRes[8];
        size_t qty = *((size_t*)qty_ptr);
        int res = 0;
        const __m128i* pVect1 = reinterpret_cast<const __m128i*>(pVect1);
        const __m128i* pVect2 = reinterpret_cast<const __m128i*>(pVect2);

        size_t qty4 = qty >> 2;
        const __m128i* pEnd1 = reinterpret_cast<const __m128i*>(pVect1+4*qty4);

        __m128i diff, v1, v2;
        __m128i sum = _mm_set1_epi32(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
            v1 = _mm_loadu_si128(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_si128 (pVect2);
            pVect2 += 4;
            diff = _mm_sub_epi32(v1, v2);
            sum = _mm_add_epi32(sum, _mm_mul_epi32(diff, diff));

        }

        _mm_store_si128((__m128i*)TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    }
#endif
    class L2SpaceI : public SpaceInterface<int> {

        DISTFUNC<int> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2SpaceI(size_t dim) {
    #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
        DISTFUNC<int> L2SqrI16x = L2SqrI16xSSE;
        #if defined(USE_AVX512)
            if (AVX512Capable()){
                L2SqrI16x =  L2SqrI16xAVX512;
                // L2SqrI4x  =  L2SqrI4xAVX512;
                }
            else if (AVXCapable()){
                L2SqrI16x = L2SqrI16xAVX;
                // L2SqrI4x  =  L2SqrI4xAVX;
                }
        #elif defined(USE_AVX)
            if (AVXCapable()){
                L2SqrI16x = L2SqrI16xAVX;
                // L2SqrI4x  =  L2SqrI4xAVX;
                }
        #endif
            if (dim % 16 == 0){
                fstdistfunc_ = L2SqrI16x;
            }
            else if(dim % 4 == 0) {
                fstdistfunc_ = L2SqrI4x;
            }
            else {
                fstdistfunc_ = L2SqrI;
            }

        #endif
            dim_ = dim;
            data_size_ = dim * sizeof(unsigned char);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<int> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2SpaceI() {}
    };


}
