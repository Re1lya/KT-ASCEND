#ifndef CPUINFER_VENDOR_VENDOR_H
#define CPUINFER_VENDOR_VENDOR_H

#include <cstdint>
#include <stdexcept>
#include <string>

#if defined(KTRANSFORMERS_USE_ASCEND)
#include "ascend.h"
#define KTRANSFORMERS_HAS_DEVICE_STREAM_CALLBACKS 1
#elif defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
#include "cuda.h"
#define KTRANSFORMERS_HAS_DEVICE_STREAM_CALLBACKS 1
#elif defined(KTRANSFORMERS_USE_ROCM)
#define __HIP_PLATFORM_AMD__
#include "hip.h"
#define KTRANSFORMERS_HAS_DEVICE_STREAM_CALLBACKS 1
#elif defined(KTRANSFORMERS_USE_MUSA)
#include "musa.h"
#define KTRANSFORMERS_HAS_DEVICE_STREAM_CALLBACKS 1
#elif defined(KTRANSFORMERS_USE_MACA)
#include "maca.h"
#define KTRANSFORMERS_HAS_DEVICE_STREAM_CALLBACKS 1
#endif

#if defined(KTRANSFORMERS_HAS_DEVICE_STREAM_CALLBACKS) && !defined(KTRANSFORMERS_USE_ASCEND)
namespace ktransformers::vendor {

using DeviceStream = cudaStream_t;
using HostFunction = cudaHostFn_t;

inline void launch_host_function(uintptr_t stream_handle, HostFunction function, void* args, const char* file,
                                 int line) {
  if (stream_handle == 0) {
    throw std::invalid_argument("device stream handle must be non-zero");
  }
  const auto status =
      cudaLaunchHostFunc(reinterpret_cast<DeviceStream>(stream_handle), static_cast<HostFunction>(function), args);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaLaunchHostFunc failed with code ") + std::to_string(status) + " at " +
                             file + ":" + std::to_string(line) + ": " + cudaGetErrorString(status));
  }
}

}  // namespace ktransformers::vendor
#endif

#if defined(KTRANSFORMERS_HAS_DEVICE_STREAM_CALLBACKS)
#define KTRANSFORMERS_VENDOR_LAUNCH_HOST_FUNCTION(stream, function, args) \
  ::ktransformers::vendor::launch_host_function((stream), (function), (args), __FILE__, __LINE__)
#endif

#endif  // CPUINFER_VENDOR_VENDOR_H
