#pragma once

#include <acl/acl_rt.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ktransformers::vendor {

using DeviceStream = aclrtStream;
using HostFunction = aclrtHostFunc;

inline void check_acl_runtime(aclError status, const char* api, const char* file, int line) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(api) + " failed with code " + std::to_string(status) + " at " + file + ":" +
                             std::to_string(line));
  }
}

inline void launch_host_function(uintptr_t stream_handle, HostFunction function, void* args, const char* file,
                                 int line) {
  if (stream_handle == 0) {
    throw std::invalid_argument("Ascend stream handle must be non-zero");
  }
  check_acl_runtime(aclrtLaunchHostFunc(reinterpret_cast<DeviceStream>(stream_handle), function, args),
                    "aclrtLaunchHostFunc", file, line);
}

}  // namespace ktransformers::vendor
