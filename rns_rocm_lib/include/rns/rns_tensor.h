#ifndef RNS_TENSOR_H
#define RNS_TENSOR_H

#include "config.h"

namespace rns {

template<typename T>
struct RnsTensor3D {
  T* data;
  int K;  // number of primes
  int B;  // batch size
  int E;  // elements per batch (e.g., m*m for matrices)

  RNS_HOST_DEVICE RnsTensor3D() : data(nullptr), K(0), B(0), E(0) {}
  
  RNS_HOST_DEVICE RnsTensor3D(T* d, int k, int b, int e) 
    : data(d), K(k), B(b), E(e) {}

  RNS_HOST_DEVICE T& operator()(int k, int b, int e) {
    return data[k * (B * E) + b * E + e];
  }

  RNS_HOST_DEVICE const T& operator()(int k, int b, int e) const {
    return data[k * (B * E) + b * E + e];
  }

  RNS_HOST_DEVICE T* ptr(int k, int b) {
    return data + k * (B * E) + b * E;
  }

  RNS_HOST_DEVICE const T* ptr(int k, int b) const {
    return data + k * (B * E) + b * E;
  }

  RNS_HOST_DEVICE int stride_k() const { return B * E; }
  RNS_HOST_DEVICE int stride_b() const { return E; }
  RNS_HOST_DEVICE size_t total_size() const { return (size_t)K * B * E; }
};

template<typename T>
struct RnsTensor2D {
  T* data;
  int K;
  int N;

  RNS_HOST_DEVICE RnsTensor2D() : data(nullptr), K(0), N(0) {}
  
  RNS_HOST_DEVICE RnsTensor2D(T* d, int k, int n) 
    : data(d), K(k), N(n) {}

  RNS_HOST_DEVICE T& operator()(int k, int n) {
    return data[k * N + n];
  }

  RNS_HOST_DEVICE const T& operator()(int k, int n) const {
    return data[k * N + n];
  }

  RNS_HOST_DEVICE size_t total_size() const { return (size_t)K * N; }
};

} // namespace rns

#endif // RNS_TENSOR_H
