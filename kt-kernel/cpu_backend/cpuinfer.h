/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-08-07 09:47:43
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CPUINFER_H
#define CPUINFER_CPUINFER_H

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "./vendors/vendor.h"
#include "llama.cpp/ggml-impl.h"
#include "task_queue.h"
#include "worker_pool.h"

class CPUInfer {
 public:
  CPUInfer(int thread_num) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }
  CPUInfer(int thread_num, int numa_id) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num, numa_id);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  CPUInfer(WorkerPoolConfig config) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(config);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  ~CPUInfer() {
    printf("CPUInfer[0x%lx]: Goodbye\n", (intptr_t)this);
    delete backend_;
    delete task_queue_;
  }

  CPUInfer(const CPUInfer&) = delete;
  CPUInfer& operator=(const CPUInfer&) = delete;
  CPUInfer(CPUInfer&&) = delete;
  CPUInfer& operator=(CPUInfer&&) = delete;

  template <typename Func, typename Obj, typename... Args>
  void enqueue(Func f, Obj* obj, Args... args) {
    task_queue_->enqueue([=]() { std::invoke(f, *obj, args...); });
  }

  void submit(std::pair<intptr_t, intptr_t> params) {
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    func(args);
  }
#if defined(KTRANSFORMERS_HAS_DEVICE_STREAM_CALLBACKS)
  struct SubmitArgs {
    CPUInfer* cpuinfer;
    void (*func)(void*);
    void* args;
  };

  static void submit_callback_(void* submit_args) noexcept {
    std::unique_ptr<SubmitArgs> state(static_cast<SubmitArgs*>(submit_args));
    try {
      state->func(state->args);
    } catch (...) {
      state->cpuinfer->record_device_callback_error_(std::current_exception());
    }
  }

  void submit_with_device_stream(uintptr_t user_device_stream, std::pair<intptr_t, intptr_t> params) {
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    auto state = std::make_unique<SubmitArgs>(SubmitArgs{this, func, args});
    KTRANSFORMERS_VENDOR_LAUNCH_HOST_FUNCTION(user_device_stream, &submit_callback_, state.get());
    state.release();
  }

  void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
    submit_with_device_stream(static_cast<uintptr_t>(user_cuda_stream), params);
  }
#endif

  struct SyncArgs {
    CPUInfer* cpuinfer;
    size_t allow_n_pending;
  };

  static void sync_(void* sync_args) {
    SyncArgs* args = static_cast<SyncArgs*>(sync_args);
    args->cpuinfer->task_queue_->sync(args->allow_n_pending);
  }

  void sync(size_t allow_n_pending = 0) {
    rethrow_device_callback_error();
    SyncArgs args{this, allow_n_pending};
    sync_(&args);
    rethrow_device_callback_error();
  }
#if defined(KTRANSFORMERS_HAS_DEVICE_STREAM_CALLBACKS)
  static void sync_callback_(void* sync_args) noexcept {
    std::unique_ptr<SyncArgs> args(static_cast<SyncArgs*>(sync_args));
    try {
      sync_(args.get());
    } catch (...) {
      args->cpuinfer->record_device_callback_error_(std::current_exception());
    }
  }

  void sync_with_device_stream(uintptr_t user_device_stream, size_t allow_n_pending = 0) {
    rethrow_device_callback_error();
    auto args = std::make_unique<SyncArgs>(SyncArgs{this, allow_n_pending});
    KTRANSFORMERS_VENDOR_LAUNCH_HOST_FUNCTION(user_device_stream, &sync_callback_, args.get());
    args.release();
  }

  void sync_with_cuda_stream(intptr_t user_cuda_stream, size_t allow_n_pending = 0) {
    sync_with_device_stream(static_cast<uintptr_t>(user_cuda_stream), allow_n_pending);
  }
#endif

  void rethrow_device_callback_error() {
    std::exception_ptr error;
    {
      std::lock_guard<std::mutex> lock(device_callback_error_mutex_);
      error = device_callback_error_;
      device_callback_error_ = nullptr;
    }
    if (error) {
      std::rethrow_exception(error);
    }
  }

 public:
  WorkerPool* backend_;
  TaskQueue* task_queue_;

 private:
  void record_device_callback_error_(std::exception_ptr error) noexcept {
    try {
      std::lock_guard<std::mutex> lock(device_callback_error_mutex_);
      if (!device_callback_error_) {
        device_callback_error_ = error;
      }
    } catch (...) {
      // A C runtime callback must never allow a C++ exception to cross the ABI boundary.
    }
  }

  std::mutex device_callback_error_mutex_;
  std::exception_ptr device_callback_error_;
};

#endif
