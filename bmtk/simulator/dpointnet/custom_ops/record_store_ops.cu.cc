// Host record store kernels: see record_store_ops.cc for the contract.
//
// Design adapted from Javier Galvan's v1_model_utils/cuda_frame_store raw mode
// (commit 5d58bc3; credit: Javier). No device code: the store only sequences
// cudaMemcpyAsync transfers on its own stream against TensorFlow's compute
// stream with events, so the library needs no per-architecture SASS.
#define EIGEN_USE_GPU

#include <cuda_runtime.h>

#include <map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

namespace {

#define RETURN_IF_CUDA_ERROR(expr)                                            \
  do {                                                                        \
    cudaError_t cuda_status = (expr);                                         \
    if (cuda_status != cudaSuccess) {                                         \
      return errors::Internal("Host record store: ", #expr, " failed: ",      \
                              cudaGetErrorString(cuda_status));               \
    }                                                                         \
  } while (0)

class HostRecordStore : public ResourceBase {
 public:
  HostRecordStore(int64 frames, const TensorShape& frame_shape, DataType dtype, int prefetch)
      : frames_(frames), frame_shape_(frame_shape), dtype_(dtype), prefetch_(prefetch),
        frame_elements_(frame_shape.num_elements()),
        frame_bytes_(frame_shape.num_elements() * DataTypeSize(dtype)),
        offloaded_(frames, nullptr) {}

  ~HostRecordStore() override {
    // Teardown errors are not reportable here (the CUDA runtime may already
    // be unloading at process exit); best effort only.
    if (copy_ != nullptr) {
      cudaSetDevice(device_);
      cudaStreamSynchronize(copy_);
    }
    resident_.clear();
    for (cudaEvent_t event : offloaded_) if (event) cudaEventDestroy(event);
    for (cudaEvent_t event : pool_) cudaEventDestroy(event);
    if (ready_) cudaEventDestroy(ready_);
    if (copy_) cudaStreamDestroy(copy_);
    if (host_) cudaFreeHost(host_);
  }

  string DebugString() const override {
    return strings::StrCat("HostRecordStore(frames=", frames_, ", frame_bytes=", frame_bytes_,
                           ", dtype=", DataTypeString(dtype_), ")");
  }

  int64 frames() const { return frames_; }
  int64 frame_elements() const { return frame_elements_; }
  int64 frame_bytes() const { return frame_bytes_; }
  const TensorShape& frame_shape() const { return frame_shape_; }
  DataType dtype() const { return dtype_; }
  int prefetch() const { return prefetch_; }

  absl::Status Offload(cudaStream_t compute, const void* source, int64 index, int64 staging_slots) {
    mutex_lock lock(mu_);
    TF_RETURN_IF_ERROR(EnsureInitialized());
    if (index < 0 || index >= frames_) {
      return errors::InvalidArgument("Host record store: offload index ", index, " outside [0, ", frames_, ")");
    }
    if (index == 0) TF_RETURN_IF_ERROR(ReleaseResidents());
    // The copy may only read the staging slot once every compute-stream
    // operation issued so far (the step's record write included) has finished.
    RETURN_IF_CUDA_ERROR(cudaEventRecord(ready_, compute));
    RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(copy_, ready_, 0));
    RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(host_ + index * frame_bytes_, source, frame_bytes_,
                                         cudaMemcpyDeviceToHost, copy_));
    if (offloaded_[index] == nullptr) {
      RETURN_IF_CUDA_ERROR(cudaEventCreateWithFlags(&offloaded_[index], cudaEventDisableTiming));
    }
    RETURN_IF_CUDA_ERROR(cudaEventRecord(offloaded_[index], copy_));
    has_frame_[index] = 1;
    // The next step writes the staging slot frame `index + 1` uses, which held
    // frame `index + 1 - staging_slots`: the device waits for that offload
    // (normally long complete) instead of the host.
    const int64 reuse = index + 1 - staging_slots;
    if (reuse >= 0 && has_frame_[reuse]) {
      RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(compute, offloaded_[reuse], 0));
    }
    return absl::OkStatus();
  }

  absl::Status Take(OpKernelContext* context, cudaStream_t compute, int64 index, int64 prefetch_index) {
    mutex_lock lock(mu_);
    TF_RETURN_IF_ERROR(EnsureInitialized());
    TF_RETURN_IF_ERROR(CheckStored(index, "take"));
    auto found = resident_.find(index);
    if (found == resident_.end()) {
      Resident resident;
      TF_RETURN_IF_ERROR(StartUpload(context, compute, index, &resident));
      found = resident_.emplace(index, std::move(resident)).first;
    }
    Resident& resident = found->second;
    if (!resident.taken) {
      RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(compute, resident.done, 0));
      pool_.push_back(resident.done);
      resident.done = nullptr;
      resident.taken = true;
    }
    context->set_output(0, resident.tensor);
    ++takes_;
    if (prefetch_index >= 0 && prefetch_index < frames_) {
      TF_RETURN_IF_ERROR(PrefetchLocked(context, compute, prefetch_index));
    }
    return absl::OkStatus();
  }

  absl::Status Prefetch(OpKernelContext* context, cudaStream_t compute, int64 index) {
    mutex_lock lock(mu_);
    TF_RETURN_IF_ERROR(EnsureInitialized());
    if (index < 0 || index >= frames_) return absl::OkStatus();
    return PrefetchLocked(context, compute, index);
  }

  void Stats(int64* out) {
    mutex_lock lock(mu_);
    int64 offloaded = 0;
    for (char flag : has_frame_) offloaded += flag;
    out[0] = frames_;
    out[1] = frame_bytes_;
    out[2] = host_ ? frames_ * frame_bytes_ : 0;
    out[3] = offloaded;
    out[4] = static_cast<int64>(resident_.size());
    out[5] = takes_;
  }

 private:
  struct Resident {
    Tensor tensor;
    cudaEvent_t done = nullptr;  // upload completion, until the compute stream has waited for it
    bool taken = false;
  };

  absl::Status EnsureInitialized() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (host_ != nullptr) return absl::OkStatus();
    RETURN_IF_CUDA_ERROR(cudaGetDevice(&device_));
    const size_t bytes = static_cast<size_t>(frames_) * static_cast<size_t>(frame_bytes_);
    void* host = nullptr;
    cudaError_t status = cudaHostAlloc(&host, bytes, cudaHostAllocPortable);
    if (status != cudaSuccess) {
      return errors::ResourceExhausted("Host record store: cudaHostAlloc of ", bytes,
                                       " pinned bytes failed: ", cudaGetErrorString(status));
    }
    host_ = static_cast<char*>(host);
    RETURN_IF_CUDA_ERROR(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
    RETURN_IF_CUDA_ERROR(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming));
    has_frame_.assign(frames_, 0);
    return absl::OkStatus();
  }

  absl::Status CheckStored(int64 index, const char* what) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (index < 0 || index >= frames_) {
      return errors::InvalidArgument("Host record store: ", what, " index ", index, " outside [0, ", frames_, ")");
    }
    if (!has_frame_[index]) {
      return errors::FailedPrecondition("Host record store: frame ", index, " was never offloaded");
    }
    return absl::OkStatus();
  }

  absl::Status PoolEvent(cudaEvent_t* event) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (!pool_.empty()) {
      *event = pool_.back();
      pool_.pop_back();
      return absl::OkStatus();
    }
    RETURN_IF_CUDA_ERROR(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
    return absl::OkStatus();
  }

  // Allocates a device frame and enqueues its upload on the copy stream; the
  // upload waits for the frame's offload and for every compute operation
  // issued so far, so the fresh allocation cannot alias live device work.
  absl::Status StartUpload(OpKernelContext* context, cudaStream_t compute, int64 index, Resident* resident)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    TF_RETURN_IF_ERROR(context->allocate_temp(dtype_, frame_shape_, &resident->tensor));
    RETURN_IF_CUDA_ERROR(cudaEventRecord(ready_, compute));
    RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(copy_, ready_, 0));
    RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(copy_, offloaded_[index], 0));
    RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(const_cast<char*>(resident->tensor.tensor_data().data()),
                                         host_ + index * frame_bytes_, frame_bytes_,
                                         cudaMemcpyHostToDevice, copy_));
    TF_RETURN_IF_ERROR(PoolEvent(&resident->done));
    RETURN_IF_CUDA_ERROR(cudaEventRecord(resident->done, copy_));
    resident->taken = false;
    return absl::OkStatus();
  }

  absl::Status PrefetchLocked(OpKernelContext* context, cudaStream_t compute, int64 index)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    TF_RETURN_IF_ERROR(CheckStored(index, "prefetch"));
    if (resident_.count(index)) return absl::OkStatus();
    // The reverse loop walks indices downwards: the largest resident index is
    // the one consumed longest ago.
    while (static_cast<int>(resident_.size()) >= prefetch_ + 2) {
      TF_RETURN_IF_ERROR(Evict(std::prev(resident_.end())));
    }
    Resident resident;
    TF_RETURN_IF_ERROR(StartUpload(context, compute, index, &resident));
    resident_.emplace(index, std::move(resident));
    return absl::OkStatus();
  }

  absl::Status Evict(std::map<int64, Resident>::iterator entry) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    Resident& resident = entry->second;
    if (!resident.taken) {
      // Never consumed: the upload may still be writing the buffer TensorFlow
      // is about to recycle, so wait for it on the host (a rare path).
      RETURN_IF_CUDA_ERROR(cudaEventSynchronize(resident.done));
      pool_.push_back(resident.done);
      resident.done = nullptr;
    }
    resident_.erase(entry);
    return absl::OkStatus();
  }

  absl::Status ReleaseResidents() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    while (!resident_.empty()) TF_RETURN_IF_ERROR(Evict(resident_.begin()));
    return absl::OkStatus();
  }

  mutex mu_;
  const int64 frames_;
  const TensorShape frame_shape_;
  const DataType dtype_;
  const int prefetch_;
  const int64 frame_elements_;
  const int64 frame_bytes_;
  int device_ = 0;
  char* host_ = nullptr;
  cudaStream_t copy_ = nullptr;
  cudaEvent_t ready_ = nullptr;
  std::vector<cudaEvent_t> offloaded_ TF_GUARDED_BY(mu_);
  std::vector<char> has_frame_ TF_GUARDED_BY(mu_);
  std::map<int64, Resident> resident_ TF_GUARDED_BY(mu_);
  std::vector<cudaEvent_t> pool_ TF_GUARDED_BY(mu_);
  int64 takes_ TF_GUARDED_BY(mu_) = 0;
};

class HostRecordStoreOp : public ResourceOpKernel<HostRecordStore> {
 public:
  explicit HostRecordStoreOp(OpKernelConstruction* context) : ResourceOpKernel<HostRecordStore>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("frames", &frames_));
    OP_REQUIRES_OK(context, context->GetAttr("frame_shape", &frame_shape_));
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("prefetch", &prefetch_));
    OP_REQUIRES(context, frame_shape_.num_elements() > 0,
                errors::InvalidArgument("Host record store frames must have a positive static size"));
  }

 private:
  absl::Status CreateResource(HostRecordStore** resource) override {
    *resource = new HostRecordStore(frames_, frame_shape_, dtype_, prefetch_);
    return absl::OkStatus();
  }

  absl::Status VerifyResource(HostRecordStore* resource) override {
    if (resource->frames() != frames_ || resource->frame_shape() != frame_shape_
        || resource->dtype() != dtype_ || resource->prefetch() != prefetch_) {
      return errors::InvalidArgument("Host record store ", resource->DebugString(),
                                     " was created with different frames, shape, dtype or prefetch");
    }
    return absl::OkStatus();
  }

  int64 frames_;
  TensorShape frame_shape_;
  DataType dtype_;
  int prefetch_;
};

REGISTER_KERNEL_BUILDER(Name("DpointnetHostRecordStore").Device(DEVICE_GPU).HostMemory("resource"),
                        HostRecordStoreOp);

absl::Status LookupStore(OpKernelContext* context, core::RefCountPtr<HostRecordStore>* store) {
  return LookupResource(context, HandleFromInput(context, 0), store);
}

int64 HostScalar(OpKernelContext* context, int input) {
  return static_cast<int64>(context->input(input).scalar<int32>()());
}

template <typename T>
class HostRecordOffloadOp : public OpKernel {
 public:
  explicit HostRecordOffloadOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("staging_slots", &staging_slots_));
  }

  void Compute(OpKernelContext* context) override {
    core::RefCountPtr<HostRecordStore> store;
    OP_REQUIRES_OK(context, LookupStore(context, &store));
    OP_REQUIRES(context, store->dtype() == DataTypeToEnum<T>::value,
                errors::InvalidArgument("Host record store dtype ", DataTypeString(store->dtype()),
                                        " does not match the offloaded frame dtype"));
    core::RefCountPtr<Var> staging;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 1), &staging));
    OP_REQUIRES(context, context->input(2).NumElements() == 1 && context->input(3).NumElements() == 1,
                errors::InvalidArgument("slot and index must be scalars"));
    const int64 slot = HostScalar(context, 2);
    const int64 index = HostScalar(context, 3);
    OP_REQUIRES(context, slot >= 0 && slot < staging_slots_,
                errors::InvalidArgument("staging slot ", slot, " outside [0, ", staging_slots_, ")"));
    const char* source = nullptr;
    {
      tf_shared_lock lock(*staging->mu());
      const Tensor* buffer = staging->tensor();
      OP_REQUIRES(context, buffer->dtype() == DataTypeToEnum<T>::value,
                  errors::InvalidArgument("staging variable dtype must match the frame dtype"));
      OP_REQUIRES(context, buffer->NumElements() == staging_slots_ * store->frame_elements(),
                  errors::InvalidArgument("staging variable must hold staging_slots frames: ",
                                          buffer->NumElements(), " elements for ", staging_slots_, " x ",
                                          store->frame_elements()));
      source = buffer->tensor_data().data() + slot * store->frame_bytes();
    }
    OP_REQUIRES_OK(context, store->Offload(context->eigen_device<GPUDevice>().stream(), source, index,
                                           staging_slots_));
    Tensor* token = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &token));
    token->scalar<int32>()() = static_cast<int32>(index);
  }

 private:
  int64 staging_slots_;
};

template <typename T>
class HostRecordTakeOp : public OpKernel {
 public:
  explicit HostRecordTakeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    core::RefCountPtr<HostRecordStore> store;
    OP_REQUIRES_OK(context, LookupStore(context, &store));
    OP_REQUIRES(context, store->dtype() == DataTypeToEnum<T>::value,
                errors::InvalidArgument("Host record store dtype ", DataTypeString(store->dtype()),
                                        " does not match the requested frame dtype"));
    OP_REQUIRES(context, context->input(1).NumElements() == 1 && context->input(2).NumElements() == 1,
                errors::InvalidArgument("index and prefetch must be scalars"));
    OP_REQUIRES_OK(context, store->Take(context, context->eigen_device<GPUDevice>().stream(),
                                        HostScalar(context, 1), HostScalar(context, 2)));
  }
};

class HostRecordPrefetchOp : public OpKernel {
 public:
  explicit HostRecordPrefetchOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    core::RefCountPtr<HostRecordStore> store;
    OP_REQUIRES_OK(context, LookupStore(context, &store));
    OP_REQUIRES(context, context->input(1).NumElements() == 1, errors::InvalidArgument("index must be a scalar"));
    const int64 index = HostScalar(context, 1);
    OP_REQUIRES_OK(context, store->Prefetch(context, context->eigen_device<GPUDevice>().stream(), index));
    Tensor* token = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &token));
    token->scalar<int32>()() = static_cast<int32>(index);
  }
};

class HostRecordStatsOp : public OpKernel {
 public:
  explicit HostRecordStatsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    core::RefCountPtr<HostRecordStore> store;
    OP_REQUIRES_OK(context, LookupStore(context, &store));
    Tensor* stats = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({6}), &stats));
    store->Stats(stats->flat<int64>().data());
  }
};

#define REGISTER_RECORD_STORE_KERNELS(T)                                                   \
  REGISTER_KERNEL_BUILDER(Name("DpointnetHostRecordOffload")                               \
                              .Device(DEVICE_GPU)                                          \
                              .TypeConstraint<T>("T")                                      \
                              .HostMemory("store")                                         \
                              .HostMemory("staging")                                       \
                              .HostMemory("slot")                                          \
                              .HostMemory("index")                                         \
                              .HostMemory("token"),                                        \
                          HostRecordOffloadOp<T>);                                         \
  REGISTER_KERNEL_BUILDER(Name("DpointnetHostRecordTake")                                  \
                              .Device(DEVICE_GPU)                                          \
                              .TypeConstraint<T>("T")                                      \
                              .HostMemory("store")                                         \
                              .HostMemory("index")                                         \
                              .HostMemory("prefetch"),                                     \
                          HostRecordTakeOp<T>);

REGISTER_RECORD_STORE_KERNELS(Eigen::half);
REGISTER_RECORD_STORE_KERNELS(float);
#undef REGISTER_RECORD_STORE_KERNELS

REGISTER_KERNEL_BUILDER(Name("DpointnetHostRecordPrefetch")
                            .Device(DEVICE_GPU)
                            .HostMemory("store")
                            .HostMemory("index")
                            .HostMemory("token"),
                        HostRecordPrefetchOp);
REGISTER_KERNEL_BUILDER(Name("DpointnetHostRecordStats")
                            .Device(DEVICE_GPU)
                            .HostMemory("store")
                            .HostMemory("stats"),
                        HostRecordStatsOp);

}  // namespace
