#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vendor.h"

#define ACL_CHECK(call) ktransformers::vendor::check_acl_runtime((call), #call, __FILE__, __LINE__)

struct CallbackState {
  std::atomic<int>* counter;
  std::atomic<int>* errors;
  int expected;
  int* observed_data;
};

// P0 callback: host state only. No CANN work, allocation, transfer, or synchronization.
static void ordered_callback(void* opaque) {
  auto* state = static_cast<CallbackState*>(opaque);
  if (state->observed_data != nullptr && *state->observed_data != state->expected) {
    state->errors->fetch_add(1, std::memory_order_relaxed);
  }
  const int sequence = state->counter->fetch_add(1, std::memory_order_relaxed);
  if (sequence != state->expected) {
    state->errors->fetch_add(1, std::memory_order_relaxed);
  }
  if (state->observed_data != nullptr) {
    *state->observed_data = state->expected + 1;
  }
}

static int run_single_callback() {
  aclrtStream stream = nullptr;
  void* device = nullptr;
  int* input = nullptr;
  int* middle = nullptr;
  int* output = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMalloc(&device, sizeof(int), ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&input), sizeof(int)));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&middle), sizeof(int)));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&output), sizeof(int)));
  *input = 0;
  *middle = -1;
  *output = -1;

  std::atomic<int> counter{0};
  std::atomic<int> errors{0};
  CallbackState state{&counter, &errors, 0, middle};
  ACL_CHECK(aclrtMemcpyAsync(device, sizeof(int), input, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK(aclrtMemcpyAsync(middle, sizeof(int), device, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST, stream));
  KTRANSFORMERS_VENDOR_LAUNCH_HOST_FUNCTION(reinterpret_cast<uintptr_t>(stream), &ordered_callback, &state);
  ACL_CHECK(aclrtMemcpyAsync(device, sizeof(int), middle, sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK(aclrtMemcpyAsync(output, sizeof(int), device, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));

  const bool passed = counter.load() == 1 && errors.load() == 0 && *middle == 1 && *output == 1;
  std::cout << "single counter=" << counter.load() << " errors=" << errors.load() << " output=" << *output
            << '\n';
  ACL_CHECK(aclrtFreeHost(output));
  ACL_CHECK(aclrtFreeHost(middle));
  ACL_CHECK(aclrtFreeHost(input));
  ACL_CHECK(aclrtFree(device));
  ACL_CHECK(aclrtDestroyStream(stream));
  return passed ? 0 : 1;
}

static int run_stress(int count) {
  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));
  std::atomic<int> counter{0};
  std::atomic<int> errors{0};
  std::vector<CallbackState> states;
  states.reserve(count);
  for (int index = 0; index < count; ++index) {
    states.push_back(CallbackState{&counter, &errors, index, nullptr});
    KTRANSFORMERS_VENDOR_LAUNCH_HOST_FUNCTION(reinterpret_cast<uintptr_t>(stream), &ordered_callback, &states.back());
  }
  ACL_CHECK(aclrtSynchronizeStream(stream));
  const bool passed = counter.load() == count && errors.load() == 0;
  std::cout << "stress launches=" << count << " counter=" << counter.load() << " errors=" << errors.load() << '\n';
  ACL_CHECK(aclrtDestroyStream(stream));
  return passed ? 0 : 1;
}

static int run_two_streams(int count) {
  aclrtStream streams[2] = {nullptr, nullptr};
  ACL_CHECK(aclrtCreateStream(&streams[0]));
  ACL_CHECK(aclrtCreateStream(&streams[1]));
  std::atomic<int> counters[2] = {0, 0};
  std::atomic<int> errors[2] = {0, 0};
  std::vector<CallbackState> states[2];
  states[0].reserve(count);
  states[1].reserve(count);
  for (int index = 0; index < count; ++index) {
    for (int stream_index = 0; stream_index < 2; ++stream_index) {
      states[stream_index].push_back(CallbackState{&counters[stream_index], &errors[stream_index], index, nullptr});
      KTRANSFORMERS_VENDOR_LAUNCH_HOST_FUNCTION(reinterpret_cast<uintptr_t>(streams[stream_index]), &ordered_callback,
                                                &states[stream_index].back());
    }
  }
  ACL_CHECK(aclrtSynchronizeStream(streams[1]));
  ACL_CHECK(aclrtSynchronizeStream(streams[0]));
  const bool passed = counters[0].load() == count && counters[1].load() == count && errors[0].load() == 0 &&
                      errors[1].load() == 0;
  std::cout << "multi_stream count_a=" << counters[0].load() << " count_b=" << counters[1].load()
            << " errors_a=" << errors[0].load() << " errors_b=" << errors[1].load() << '\n';
  ACL_CHECK(aclrtDestroyStream(streams[1]));
  ACL_CHECK(aclrtDestroyStream(streams[0]));
  return passed ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc != 2) return 64;
  ACL_CHECK(aclInit(nullptr));
  ACL_CHECK(aclrtSetDevice(0));
  int result = 65;
  const std::string mode = argv[1];
  if (mode == "single") result = run_single_callback();
  if (mode == "stress") result = run_stress(10000);
  if (mode == "multi") result = run_two_streams(1000);
  ACL_CHECK(aclrtResetDevice(0));
  ACL_CHECK(aclFinalize());
  return result;
}
