#pragma once
#include "hnswlib.h"

namespace hnswlib {

    static float
    InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        float res = 0;
        for (unsigned i = 0; i < qty; i++) {
            res += ((float *) pVect1)[i] * ((float *) pVect2)[i];
        }
        return res;

    }

    static float
    InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
        return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
    }

#if defined(USE_AVX)

// Favor using AVX if available.
    static float
    InnerProductSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        size_t qty16 = qty / 16;
        size_t qty4 = qty / 4;

        const float *pEnd1 = pVect1 + 16 * qty16;
        const float *pEnd2 = pVect1 + 4 * qty4;

        __m256 sum256 = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m256 v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            __m256 v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
        }

        __m128 v1, v2;
        __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

        while (pVect1 < pEnd2) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }

        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];;
        return sum;
    }
    
    static float
    InnerProductDistanceSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        return 1.0f - InnerProductSIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
    }

#endif

//  add InnerProductSIMD4ExtAVX with avx512
#if defined(USE_AVX512)

    static float
    InnerProductSIMD4ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        size_t qty16 = qty / 16;
        size_t qty4 = qty / 4;

        const float *pEnd1 = pVect1 + 16 * qty16;
        const float *pEnd2 = pVect1 + 4 * qty4;

        __m512 sum512 = _mm512_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m512 v1 = _mm512_loadu_ps(pVect1);
            pVect1 += 16;
            __m512 v2 = _mm512_loadu_ps(pVect2);
            pVect2 += 16;
            sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
        }

        __m256 v1, v2;
        __m256 sum_prod = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));

        while (pVect1 < pEnd2) {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm256_add_ps(sum_prod, _mm256_mul_ps(v1, v2));
        }

        _mm256_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];;
        return sum;
    }
    
    static float
    InnerProductDistanceSIMD4ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        return 1.0f - InnerProductSIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
    }

#endif

#if defined(USE_SSE)

    static float
    InnerProductSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        size_t qty16 = qty / 16;
        size_t qty4 = qty / 4;

        const float *pEnd1 = pVect1 + 16 * qty16;
        const float *pEnd2 = pVect1 + 4 * qty4;

        __m128 v1, v2;
        __m128 sum_prod = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }

        while (pVect1 < pEnd2) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }

        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return sum;
    }

    static float
    InnerProductDistanceSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        return 1.0f - InnerProductSIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
    }

#endif


#if defined(USE_AVX512)

    static float
    InnerProductSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) { //512/sizeof(float) =16
        float PORTABLE_ALIGN64 TmpRes[16];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        size_t qty16 = qty / 16;


        const float *pEnd1 = pVect1 + 16 * qty16;

        __m512 sum512 = _mm512_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m512 v1 = _mm512_loadu_ps(pVect1);
            pVect1 += 16;
            __m512 v2 = _mm512_loadu_ps(pVect2);
            pVect2 += 16;
            sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
        }

        _mm512_store_ps(TmpRes, sum512);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] + TmpRes[15];

        return sum;
    }

    static float
    InnerProductDistanceSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        return 1.0f - InnerProductSIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
    }

#endif

#if defined(USE_AVX)

    static float
    InnerProductSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8]; //256/sizeof(float) =8
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        size_t qty16 = qty / 16;


        const float *pEnd1 = pVect1 + 16 * qty16;

        __m256 sum256 = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m256 v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            __m256 v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
        }

        _mm256_store_ps(TmpRes, sum256);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

        return sum;
    }

    static float
    InnerProductDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        return 1.0f - InnerProductSIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
    }

#endif

#if defined(USE_SSE)

    static float
    InnerProductSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        size_t qty16 = qty / 16;

        const float *pEnd1 = pVect1 + 16 * qty16;

        __m128 v1, v2;
        __m128 sum_prod = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }
        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return sum;
    }

    static float
    InnerProductDistanceSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        return 1.0f - InnerProductSIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
    }

#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    DISTFUNC<float> InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
    DISTFUNC<float> InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
    DISTFUNC<float> InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
    DISTFUNC<float> InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;

    static float
    InnerProductDistanceSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
        float *pVect1 = (float *) pVect1v + qty16;
        float *pVect2 = (float *) pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
        return 1.0f - (res + res_tail);
    }

    static float
    InnerProductDistanceSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
        size_t qty_left = qty - qty4;

        float *pVect1 = (float *) pVect1v + qty4;
        float *pVect2 = (float *) pVect2v + qty4;
        float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

        return 1.0f - (res + res_tail);
    }
#endif

    class InnerProductSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        InnerProductSpace(size_t dim) {
            fstdistfunc_ = InnerProductDistance;
    #if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
        #if defined(USE_AVX512)
            if (AVX512Capable()) {
                InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
                InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
            } else if (AVXCapable()) {
                InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
                InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
            }
        #elif defined(USE_AVX)
            if (AVXCapable()) {
                InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
                InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
            }
        #endif
        #if defined(USE_AVX)
            if (AVXCapable()) {
                InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
                InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
            }
        #endif

            if (dim % 16 == 0)
                fstdistfunc_ = InnerProductDistanceSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = InnerProductDistanceSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
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

    ~InnerProductSpace() {}
    };





#define AVX512_ALIGN            alignas(64)
#define AVX512_TOTAL_INT16      (64 / sizeof(int16_t))
#define AVX512_TOTAL_INT32      (64 / sizeof(int32_t))
#if defined(USE_VNNI)
// static int IPdistanceI(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr){
//         size_t qty = *((size_t *) qty_ptr);
//         int res = 0;
//         unsigned char *a = (unsigned char *) pVect1;
//         unsigned char *b = (unsigned char *) pVect2;
//         __m512i v1 = _mm512_loadu_si512(a);
//         __m512i v2 = _mm512_loadu_si512(b);

//         __m512i vresult = _mm512_set1_epi32(0);
//         vresult = _mm512_dpbusds_epi32(vresult, v1, v2);
//         _mm512_storeu_si512((void*)res, vresult);
//  return (1.0f - res);
// }


static int IPdistanceIVNNI(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr){
        int PORTABLE_ALIGN32 TmpRes[8];
        size_t qty = *((size_t *) qty_ptr);
        // int res = 0;
        // unsigned char *a = (unsigned char *) pVect1;
        // unsigned char *b = (unsigned char *) pVect2;
        size_t qty = *((size_t *) qty_ptr);

        size_t qty16 = qty / 16;

        const __m512i* pVect1 = reinterpret_cast<const __m512i*>(pVect1);
        const __m512i* pVect2 = reinterpret_cast<const __m512i*>(pVect2);
        const __m512i* pEnd1 = reinterpret_cast<const __m512i*>(pVect1+16*qty16);

        // const float *pEnd1 = pVect1 + 16 * qty16;

        __m512i sum512 = _mm512_setzero_si512();

        while (pVect1 < pEnd1) {

            __m512i v1 = _mm512_loadu_si512(pVect1);
            pVect1 += 16;
            __m512i v2 = _mm512_loadu_si512(pVect2);
            pVect2 += 16;

            __m512i vresult = _mm512_set1_epi32(0);
            vresult = _mm512_dpbusds_epi32(vresult, v1, v2);
        }

        _mm512_storeu_si512(TmpRes, vresult);
        int sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] + TmpRes[15];

        return 1.0f - sum;
}

static int IPdistanceIVNNI2(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr){
    size_t qty = *((size_t *) qty_ptr);
    int res = 0;
    unsigned char *a = (unsigned char *) pVect1;
    unsigned char *b = (unsigned char *) pVect2;
    __m512i msum512 = _mm512_setzero_si512();
    __m512i mbias512 = _mm512_set1_epi8(128);
    while(qty >= 64) {
        __m512i ma = _mm512_loadu_si512((const __m512i_u*)a);
        a += 64;
        __m512i mb = _mm512_add_epi8(mbias512, _mm512_loadu_si512((const __m512i_u*)b));
        b += 64;
        msum512 = _mm512_dpbusd_epi32(msum512, mb, ma);
        qty -= 64;
    }
    if(qty == 0) {
        return _mm512_reduce_add_epi32(msum512);
    }
    __m256i msum256 = _mm512_extracti32x8_epi32(msum512, 1);
    msum256 = _mm256_add_epi32(msum256, _mm512_extracti32x8_epi32(msum512, 0));

    __m256i mbias256 = _mm256_set1_epi8(128);
    if(qty >= 32) {
        __m256i ma = _mm256_loadu_si256((const __m256i_u*)a);
        a += 32;
        __m256i mb = _mm256_add_epi8(mbias256, _mm256_loadu_si256((const __m256i_u*)b));
        b += 32;
        msum256 = _mm256_dpbusd_epi32(msum256, mb, ma);
        qty -= 32;
    }
    __m128i msum128 = _mm256_extracti128_si256(msum256, 1);
    msum128 = _mm_add_epi32(msum128, _mm256_extracti128_si256(msum256, 0));
    __m128i mbias128 = _mm_set1_epi8(128);
    if(qty >= 16) {
        __m128i ma = _mm_loadu_si128((const __m128i_u*)a);
        a += 16;
        __m128i mb = _mm_add_epi8(mbias128, _mm_loadu_si128((const __m128i_u*)b));
        b += 16;
        msum128 = _mm_dpbusd_epi32(msum128, mb, ma);
        qty -= 16;
    }
    msum128 = _mm_hadd_epi32(msum128, msum128);
    msum128 = _mm_hadd_epi32(msum128, msum128);
    int sum = _mm_cvtsi128_si32(msum128);
    return qty ? sum + IPdistanceIRef(a, b, qty) : sum;

}

static int IPdistanceI4xVNNI(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr)
{
        size_t qty = *((size_t *) qty_ptr);
        int res = 0;
        unsigned char *a = (unsigned char *) pVect1;
        unsigned char *b = (unsigned char *) pVect2;
        qty = qty >> 2;
        for(size_t i=0;i<qty;++i){
            asm volatile (
            "vmovdqu64 %0, %%zmm0 \r\n"
            "vmovdqu64 %1, %%zmm1 \r\n"
            "vmovdqa64 %2, %%zmm2 \r\n"
            "vpdpwssd %%zmm0, %%zmm1, %%zmm2 \r\n"
            "vmovdqa32 %%zmm2, %2 \r\n"
            : : "m"(*a), "m"(*b), "m"(sums) : "zmm0", "zmm1", "zmm2");

        a += 4;
        b += 4;
        qty -= 4;
        }

    return (1.0f - sums);

}

#elif defined(USE_AVX512)
// static int IPdisntanceI(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr){
//         size_t qty = *((size_t *) qty_ptr);
//         int res = 0;
//         unsigned char *a = (unsigned char *) pVect1;
//         unsigned char *b = (unsigned char *) pVect2;
//         __m512i v1 = _mm512_loadu_si512(a);
//         __m512i v2 = _mm512_loadu_si512(b);
//         __m512i vresult = _mm512_set1_epi32(0);
//         // __m512i vresult_1 = _mm512_set1_ps(0);
//         // vresult = _mm512_dpbusds_epi32(vresult, v1, v2);
//         vresult = _mm512_maddubs_epi32(a, b);
//         // __512i vresult = _mm512_madd_epi16(vresut_1, vresult);
//         vresult = _mm512_add_epi32(vresult,_mm512_madd_epi32(a,b));
//         _mm512_storeu_si512((void*)res, vresult);
//  return (1.0f - res);
// }


// static int IPdistanceI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr){
//         size_t qty = *((size_t *) qty_ptr);
//         int res = 0;
//         unsigned char *a = (unsigned char *) pVect1;
//         unsigned char *b = (unsigned char *) pVect2;

//         size_t qty = qty / 4;
//         __m512 vresult = _mm512_set1_epi32(0);
//         for (size_t i = 0; i < qty; i++) {

//             //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

//             __m512 v1 = _mm512_loadu_si512(pVect1);
//             pVect1 += 16;
//             __m512 v2 = _mm512_loadu_si512(pVect2);
//             pVect2 += 16;
//             vresult = _mm512_add_epi32(vresult, _mm512_mul_ps(v1, v2));
//             v1 += 16;
//             v2 += 16;
//             qty -= 16;
//         }
//         _mm512_storeu_si512((void*)res, vresult);
//         return 1.0f - res;

// }


#endif
    static int
    IPdistanceI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {

        size_t qty = *((size_t *) qty_ptr);
        int res = 0;
        unsigned char *a = (unsigned char *) pVect1;
        unsigned char *b = (unsigned char *) pVect2;

        qty = qty >> 2;
        for (size_t i = 0; i < qty; i++) {

            res += ((*a)*(*b));
            a++;
            b++;
            res += ((*a)*(*b));
            a++;
            b++;
            res += ((*a)*(*b));
            a++;
            b++;
            res += ((*a)*(*b));
            a++;
            b++;
        }
        return (1.0f - res);
    }

    static int IPdistanceI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
        size_t qty = *((size_t*)qty_ptr);
        int res = 0;
        unsigned char* a = (unsigned char*)pVect1;
        unsigned char* b = (unsigned char*)pVect2;

        for(size_t i = 0; i < qty; i++)
        {
            res += ((*a)*(*b));
            a++;
            b++;
        }
        return (1.0f - res);
    }


    static int IPdistanceIRef(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
        size_t qty = *((size_t*)qty_ptr);
        int res = 0;
        unsigned char* a = (unsigned char*)pVect1;
        unsigned char* b = (unsigned char*)pVect2;

        for(size_t i = 0; i < qty; i++)
        {
            res += ((*a)*(*b));
            a++;
            b++;
        }
        return (1.0f - res);
    }
    class InnerProductSpaceI : public SpaceInterface<int> {

        DISTFUNC<int> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        InnerProductSpaceI(size_t dim) {
            if(dim % 4 == 0) {
                fstdistfunc_ = IPdistanceI4x;
            }
            else {
                fstdistfunc_ = IPdistanceI;
            }
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

        ~InnerProductSpaceI() {}
    };


}