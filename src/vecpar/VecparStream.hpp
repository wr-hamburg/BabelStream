#pragma once

#include <iostream>

#include <vecmem/containers/vector.hpp>
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "Stream.h"

#if defined(VECPAR_GPU)
    #if defined(NATIVE)
        #include "cuda.h"
        #include <vecmem/memory/cuda/managed_memory_resource.hpp>
        #include <vecmem/memory/cuda/device_memory_resource.hpp>
        #include <vecmem/containers/data/vector_buffer.hpp>
        #include <vecmem/utils/cuda/copy.hpp>
        #include <vecpar/cuda/cuda_parallelization.hpp>
    #else
        #include <vecpar/ompt/ompt_parallelization.hpp>
    #endif
#endif

#include <vecpar/all/main.hpp>

//backend = NATIVE/ompt, memory = default/managed, offload=0/1
#if defined(NATIVE) and defined(DEFAULT) and defined(VECPAR_GPU)
    #define IMPLEMENTATION_STRING "vecpar_cuda_hostdevice"
    #undef SINGLE_SOURCE
#elif defined(NATIVE) and defined(DEFAULT) and !defined(VECPAR_GPU)
    #define IMPLEMENTATION_STRING "vecpar_omp_hostmemory"
    #define SINGLE_SOURCE 1
#elif defined(NATIVE) and defined(MANAGED) and defined(VECPAR_GPU)
    #define IMPLEMENTATION_STRING "vecpar_cuda_singlesource_managedmemory"
    #define SINGLE_SOURCE 1
#elif defined(NATIVE) and defined(MANAGED) and !defined(VECPAR_GPU)
    #define IMPLEMENTATION_STRING "vecpar_omp_singlesource_managedmemory"
    #define SINGLE_SOURCE 1
#elif defined(OMPT) and defined(DEFAULT) and defined(VECPAR_GPU)
    #define IMPLEMENTATION_STRING "vecpar_ompt_gpu_singlesource_hostdevice"
    #define SINGLE_SOURCE 1
#elif defined(OMPT) and defined(DEFAULT) and !defined(VECPAR_GPU)
    #define IMPLEMENTATION_STRING "vecpar_ompt_cpu_singlesource_hostmemory"
    #define SINGLE_SOURCE 1
#else
    #define IMPLEMENTATION_STRING "NOT_RELEVANT"
 #endif

template <class T>
class VecparStream : public Stream<T>
{
protected:
    // Size of arrays
    int array_size;

    // Host side pointers or managed memory
    vecmem::vector<T> *a;
    vecmem::vector<T> *b;
    vecmem::vector<T> *c;

#if defined(VECPAR_GPU) && defined(MANAGED)
    vecmem::cuda::managed_memory_resource memoryResource;
#elif defined(VECPAR_GPU) && defined(DEFAULT)
    #if defined(NATIVE)
        vecmem::host_memory_resource memoryResource;
        vecmem::cuda::device_memory_resource dev_mem;
        vecmem::cuda::copy copy_tool;

        vecmem::data::vector_buffer<T> d_a;
        vecmem::data::vector_buffer<T> d_b;
        vecmem::data::vector_buffer<T> d_c;
    #else
        vecmem::host_memory_resource memoryResource;
        T* d_a;
        T* d_b;
        T* d_c;
    #endif
#else
    vecmem::host_memory_resource memoryResource;
#endif

public:
    VecparStream(const int, int);
    ~VecparStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};

/// define vecpar algorithms
    template <class T>
    struct vecpar_triad :
        public vecpar::algorithm::parallelizable_mmap<
            vecpar::collection::Three,
            vecmem::vector<T>, // a
            vecmem::vector<T>, // b
            vecmem::vector<T>, // c
            const T // scalar
            > {
        TARGET T& mapping_function(T& a_i, const T& b_i, const T& c_i, const T scalar) const {
            a_i = b_i + scalar * c_i;
            return a_i;
        }
    };

template <class T>
struct vecpar_add :
        public vecpar::algorithm::parallelizable_mmap<
                vecpar::collection::Three,
                vecmem::vector<T>, // c
                vecmem::vector<T>, // a
                vecmem::vector<T>> // b
                {
    TARGET T& mapping_function(T& c_i, const T& a_i, const T& b_i) const {
        c_i = a_i + b_i ;
        return c_i;
    }
};

template <class T>
struct vecpar_mul:
        public vecpar::algorithm::parallelizable_mmap<
                vecpar::collection::Two,
                vecmem::vector<T>, // b
                vecmem::vector<T>, // c
                const T > //  scalar
            {
    TARGET T& mapping_function(T& b_i, const T& c_i, const T scalar) const {
        b_i = scalar * c_i ;
        return b_i;
    }
};

template <class T>
struct vecpar_copy:
        public vecpar::algorithm::parallelizable_mmap<
                vecpar::collection::Two,
                vecmem::vector<T>, // c
                vecmem::vector<T>> // a
{
    TARGET T& mapping_function(T& c_i, const T& a_i) const {
        c_i = a_i;
        return c_i;
    }
};

template <class T>
struct vecpar_dot:
        public vecpar::algorithm::parallelizable_map_reduce<
                vecpar::collection::Two,
                T, // reduction result
                vecmem::vector<T>, // map result
                vecmem::vector<T>, // a
                vecmem::vector<T>> // b
{
    TARGET T& mapping_function(T& result, T& a_i, const T& b_i) const {
        result = a_i * b_i;
        return result;
    }

    TARGET T* reducing_function(T* result, T& crt) const {
        *result += crt;
        return result;
    }
};

template <class T>
struct vecpar_nstream : public vecpar::algorithm::parallelizable_mmap<
        vecpar::collection::Three,
        vecmem::vector<T>, // a
        vecmem::vector<T>, // b
        vecmem::vector<T>, // c
        const T> // scalar
{
    TARGET T& mapping_function(T& a_i, const T& b_i, const T& c_i, const T scalar) const {
        a_i += b_i + scalar * c_i;
        return a_i;
    }

};


