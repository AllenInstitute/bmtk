// Adaptive routing for the combined CSR-master sweep (use_adaptive_routing,
// schema adaptive_segment_ticket_v1).
//
// Independent implementation of a compressed-segment / adaptive-scheduler
// projection design, included inside namespace tensorflow by
// csr_spike_ops.cu.cc after the combined-sweep helpers it reuses.
//
// * Compressed per-row tile segments, built natively on first use for each
//   (connectivity allocation, tile width): segment_splits[row] .. [row + 1]
//   index the row's nonempty tile segments; segment_starts[s] is the first CSR
//   position of segment s and segment_starts[S] == n_edges, so a segment ends
//   where the next begins (CSR rows are contiguous). Up to 64 tiles a per-row
//   64-bit mask names the row's tiles and the segment rank is a popcount, so
//   no per-segment tile ids are stored; above 64 tiles the row's sorted tile
//   ids are binary-searched.
// * Tables are shared through a process registry of weak references keyed by
//   the actual metadata allocation, connectivity dimensions and tile width.
//   Each operator keeps strong references to at most one table per source per
//   shape (wide/narrow) plus one target-major gather table per gathered
//   source, so the registry never grows beyond live operators. Entries keep
//   the metadata tensor alive, which prevents address reuse while referenced.
// * Each launch selects its tile width from the actual workload: the
//   configured (wide) tile unless the wide grid would fill at most half of the
//   device's resident block slots, in which case a narrow tile (a quarter of
//   the wide one, clamped to 256..1024) spreads the work over more blocks.
// * Active rows come from an ordered, device-built compaction of each
//   source's [batch, width] activity plus per-sample starts; no host round
//   trip. Target-unique sources are consumed by warps that claim adaptive
//   batches of 4-32 rows from a per-block atomic ticket, compact the rows that
//   have a segment in the block's tile and walk their concatenated edges with
//   several independent loads per lane. Target-repeating or high-fanout
//   sources use block rounds: one packed block scan compacts the found rows
//   and equal-target runs are summed in registers before the shared atomics.
//   Eligible low in-degree sources keep the existing target-major gather.
//
// Arithmetic per edge is unchanged: fmul_rn(spike, weight) times each basis
// coefficient, accumulated in FP32 shared tiles and rounded once to T.
// Only the order of the FP32 atomic additions differs.
#pragma once

// Warp-batch tuning: rows per batch target about kAdaptiveBatchShare batches
// per warp; kAdaptiveRecurrentEdges independent edges per lane per walk step.
#ifndef DPOINTNET_ADAPTIVE_BATCH_SHARE
#define DPOINTNET_ADAPTIVE_BATCH_SHARE 8
#endif
#ifndef DPOINTNET_ADAPTIVE_REC_EDGES
#define DPOINTNET_ADAPTIVE_REC_EDGES 2
#endif
constexpr unsigned int kAdaptiveBatchShare = DPOINTNET_ADAPTIVE_BATCH_SHARE;
constexpr int kAdaptiveRecurrentEdges = DPOINTNET_ADAPTIVE_REC_EDGES;

enum DpointnetRoutingKind : int {
  kRoutingDenseTiles = 0,
  kRoutingSegments = 1,
  kRoutingMasks = 2,
  kRoutingCsc = 3,
  kRoutingQueue = 4,
  kRoutingSelection = 5,
};

enum DpointnetAdaptiveTraversal : int {
  kAdaptiveWarpBatches = 1,
  kAdaptiveBlockRounds = 2,
  kAdaptiveGather = 3,
};

constexpr int kRoutingCensusFields = 16;

struct DpointnetRoutingRecord {
  const void* metadata = nullptr;
  int kind = 0;
  int64_t bytes = 0;
  int64_t count = 0;
  int64_t aux = 0;
};

// Process-wide bookkeeping of live routing allocations. It records sizes only;
// it never owns, reads or copies a table. Owners erase their records on
// replacement and destruction.
class DpointnetRoutingLedger {
 public:
  static DpointnetRoutingLedger& Get() {
    static DpointnetRoutingLedger* ledger = new DpointnetRoutingLedger;
    return *ledger;
  }

  void Set(const void* owner, int slot, const DpointnetRoutingRecord& record) {
    mutex_lock lock(mu_);
    records_[{owner, slot}] = record;
  }

  void Erase(const void* owner, int slot) {
    mutex_lock lock(mu_);
    records_.erase({owner, slot});
  }

  void EraseOwner(const void* owner) {
    mutex_lock lock(mu_);
    auto it = records_.lower_bound({owner, std::numeric_limits<int>::min()});
    while (it != records_.end() && it->first.first == owner) it = records_.erase(it);
  }

  // Fields: 0 dense tile bytes, 1 dense tables, 2 segment bytes (splits and
  // starts and tile ids, masks excluded), 3 segment tables, 4 mask bytes,
  // 5 gather (CSC) bytes, 6 gather tables, 7 last-launch transient queue
  // bytes summed over operators, 8 operators with a queue record, 9 last
  // selected tile (maximum over operators), 10 traversal code (maximum),
  // 11 total segments, 12 OR of built tile-width bits, 13 adaptive operators,
  // 14-15 reserved zero.
  void Census(const void* metadata, int64_t* fields) {
    for (int field = 0; field < kRoutingCensusFields; ++field) fields[field] = 0;
    mutex_lock lock(mu_);
    for (const auto& item : records_) {
      const DpointnetRoutingRecord& record = item.second;
      if (record.metadata != metadata) continue;
      switch (record.kind) {
        case kRoutingDenseTiles:
          fields[0] += record.bytes;
          fields[1] += record.count;
          break;
        case kRoutingSegments:
          fields[2] += record.bytes;
          fields[3] += 1;
          fields[11] += record.count;
          fields[12] |= record.aux;
          break;
        case kRoutingMasks:
          fields[4] += record.bytes;
          break;
        case kRoutingCsc:
          fields[5] += record.bytes;
          fields[6] += record.count;
          break;
        case kRoutingQueue:
          fields[7] += record.bytes;
          fields[8] += 1;
          break;
        case kRoutingSelection:
          fields[9] = std::max<int64_t>(fields[9], record.bytes);
          fields[10] = std::max<int64_t>(fields[10], record.aux);
          fields[13] += 1;
          break;
      }
    }
  }

 private:
  mutex mu_;
  std::map<std::pair<const void*, int>, DpointnetRoutingRecord> records_;
};

inline int AdaptiveTileBit(int tile) {
  switch (tile) {
    case 256: return 1;
    case 512: return 2;
    case 1024: return 4;
    case 2048: return 8;
    case 3072: return 16;
    case 4096: return 32;
    case 5120: return 64;
  }
  return 0;
}

// The narrow tile paired with a wide one: a quarter of it, clamped to the
// statically sized shapes 256..1024. Equal to `wide` when no narrower shape
// exists.
inline int AdaptiveNarrowTile(int wide) {
  const int quarter = wide / 4;
  if (quarter >= 1024) return 1024;
  if (quarter >= 512) return 512;
  return 256;
}

template <typename Index>
struct AdaptiveSegmentTable {
  const DeviceBase* device = nullptr;
  Tensor metadata;
  int64_t metadata_elements = 0;
  int width = 0;
  int64_t edges = 0;
  int n_post = 0;
  int tile = 0;
  bool uint8_types = false;
  Tensor splits;   // Index [width + 1]
  Tensor starts;   // Index [segments + 1]
  Tensor tiles;    // uint32 bits in int32 [segments]; absent with masks
  Tensor masks;    // uint64 bits in int64 [width]; only up to 64 tiles
  int64_t segments = 0;
  ~AdaptiveSegmentTable() { DpointnetRoutingLedger::Get().EraseOwner(this); }
};

template <typename Index>
struct AdaptiveCscTable {
  const DeviceBase* device = nullptr;
  Tensor metadata;
  int64_t metadata_elements = 0;
  int width = 0;
  int64_t edges = 0;
  int n_post = 0;
  bool uint8_types = false;
  Tensor splits;
  Tensor targets;
  Tensor edge_positions;
  Tensor sources;
  ~AdaptiveCscTable() { DpointnetRoutingLedger::Get().EraseOwner(this); }
};

template <typename Table>
struct AdaptiveRegistry {
  mutex mu;
  std::vector<std::weak_ptr<Table>> tables;
};

template <typename Table>
AdaptiveRegistry<Table>& GetAdaptiveRegistry() {
  static AdaptiveRegistry<Table>* registry = new AdaptiveRegistry<Table>;
  return *registry;
}

template <typename Table>
bool AdaptiveTableMatches(const Table& table, const DeviceBase* device, const Tensor& metadata,
                          int width, int64_t edges, int n_post, bool uint8_types) {
  return table.device == device &&
         table.metadata.tensor_data().data() == metadata.tensor_data().data() &&
         table.metadata_elements == metadata.NumElements() && table.width == width &&
         table.edges == edges && table.n_post == n_post && table.uint8_types == uint8_types;
}

// One warp per row counts (kFill false) or writes (kFill true) the row's tile
// segments; a segment begins at the row's first edge and wherever the tile of
// the target-sorted postsynaptic id changes.
template <typename Index, bool kFill>
__global__ void AdaptiveSegmentsKernel(
    int width, int tile_targets, const Index* splits, const Index* posts, Index* segment_splits,
    Index* segment_starts, unsigned int* segment_tiles, unsigned long long* masks) {
  const int lane = threadIdx.x & 31;
  const int64_t warps = static_cast<int64_t>(gridDim.x) * (blockDim.x >> 5);
  for (int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
       row < width; row += warps) {
    const Index start = splits[row];
    const Index end = splits[row + 1];
    Index written = kFill ? segment_splits[row] : 0;
    unsigned int previous = 0xffffffffu;
    unsigned long long mask = 0;
    for (Index base = start; base < end; base += 32) {
      const Index csr = base + lane;
      const bool valid = csr < end;
      const unsigned int tile =
          valid ? static_cast<unsigned int>(static_cast<int64_t>(posts[csr]) / tile_targets)
                : 0xfffffffeu;
      unsigned int before = __shfl_up_sync(0xffffffffu, tile, 1);
      if (lane == 0) before = previous;
      const bool begins = valid && tile != before;
      const unsigned int flags = __ballot_sync(0xffffffffu, begins);
      if (kFill && begins) {
        const Index position = written + __popc(flags & ((1u << lane) - 1u));
        segment_starts[position] = csr;
        if (segment_tiles != nullptr) segment_tiles[position] = tile;
      }
      if (begins && tile < 64) mask |= 1ull << tile;
      written += __popc(flags);
      previous = __shfl_sync(0xffffffffu, tile, 31);
    }
    if constexpr (!kFill) {
      if (lane == 0) segment_splits[row] = written;
    } else if (masks != nullptr) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        mask |= __shfl_xor_sync(0xffffffffu, mask, offset);
      }
      if (lane == 0) masks[row] = mask;
    }
  }
}

template <typename Index>
__global__ void AdaptiveSetValueKernel(Index* address, Index value) {
  *address = value;
}

template <typename Index>
Status BuildAdaptiveSegments(OpKernelContext* context, const Index* posts, const Index* splits,
                             AdaptiveSegmentTable<Index>* table) {
  const GPUDevice& device = context->eigen_device<GPUDevice>();
  auto stream = device.stream();
  const auto type = DataTypeToEnum<Index>::v();
  const int width = table->width;
  const int tiles = (table->n_post + table->tile - 1) / table->tile;
  const bool use_masks = tiles <= 64;
  TF_RETURN_IF_ERROR(context->allocate_temp(type, TensorShape({width + 1}), &table->splits));
  Index* segment_splits = table->splits.template flat<Index>().data();
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>((static_cast<int64_t>(width) * 32 + kThreads - 1) / kThreads, 65535)));
  TF_RETURN_IF_ERROR(GpuLaunchKernel(AdaptiveSetValueKernel<Index>, 1, 1, 0, stream,
                                     segment_splits + width, Index(0)));
  if (width > 0) {
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        AdaptiveSegmentsKernel<Index, false>, blocks, kThreads, 0, stream, width,
        table->tile, splits, posts, segment_splits, static_cast<Index*>(nullptr),
        static_cast<unsigned int*>(nullptr), static_cast<unsigned long long*>(nullptr)));
  }
  size_t scratch_bytes = 0;
  if (cub::DeviceScan::ExclusiveSum(nullptr, scratch_bytes, segment_splits, segment_splits,
                                    width + 1, stream) != cudaSuccess) {
    cudaGetLastError();
    return errors::Internal("sizing the adaptive routing segment scan failed");
  }
  Tensor scratch;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(std::max<size_t>(scratch_bytes, 1))}), &scratch));
  if (cub::DeviceScan::ExclusiveSum(scratch.flat<int8>().data(), scratch_bytes, segment_splits,
                                    segment_splits, width + 1, stream) != cudaSuccess) {
    cudaGetLastError();
    return errors::Internal("scanning adaptive routing segment counts failed");
  }
  // One-time table construction reads the segment total on the host; warm
  // launches never synchronize or copy routing data to the host.
  Index total = 0;
  cudaMemcpyAsync(&total, segment_splits + width, sizeof(Index), cudaMemcpyDeviceToHost, stream);
  const cudaError_t counted = cudaStreamSynchronize(stream);
  if (counted != cudaSuccess) {
    return errors::Internal("counting adaptive routing segments failed: ",
                            cudaGetErrorString(counted));
  }
  table->segments = static_cast<int64_t>(total);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      type, TensorShape({table->segments + 1}), &table->starts));
  Index* starts = table->starts.template flat<Index>().data();
  TF_RETURN_IF_ERROR(GpuLaunchKernel(AdaptiveSetValueKernel<Index>, 1, 1, 0, stream,
                                     starts + table->segments, static_cast<Index>(table->edges)));
  unsigned int* segment_tiles = nullptr;
  unsigned long long* masks = nullptr;
  table->tiles = Tensor();
  table->masks = Tensor();
  if (use_masks) {
    TF_RETURN_IF_ERROR(context->allocate_temp(DT_INT64, TensorShape({width}), &table->masks));
    masks = reinterpret_cast<unsigned long long*>(table->masks.template flat<int64>().data());
  } else {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_INT32, TensorShape({table->segments}), &table->tiles));
    segment_tiles = reinterpret_cast<unsigned int*>(table->tiles.template flat<int32>().data());
  }
  if (width > 0) {
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        AdaptiveSegmentsKernel<Index, true>, blocks, kThreads, 0, stream, width, table->tile,
        splits, posts, segment_splits, starts, segment_tiles, masks));
  }
  const cudaError_t filled = cudaStreamSynchronize(stream);
  if (filled != cudaSuccess) {
    return errors::Internal("filling adaptive routing segments failed: ",
                            cudaGetErrorString(filled));
  }
  const int64_t index_bytes = static_cast<int64_t>(table->splits.TotalBytes()) +
                              static_cast<int64_t>(table->starts.TotalBytes()) +
                              static_cast<int64_t>(table->tiles.TotalBytes());
  DpointnetRoutingLedger::Get().Set(
      table, kRoutingSegments,
      {table->metadata.tensor_data().data(), kRoutingSegments, index_bytes, table->segments,
       AdaptiveTileBit(table->tile)});
  DpointnetRoutingLedger::Get().Set(
      table, kRoutingMasks,
      {table->metadata.tensor_data().data(), kRoutingMasks,
       static_cast<int64_t>(table->masks.TotalBytes()), 0, 0});
  return OkStatus();
}

template <typename Index>
Status BuildAdaptiveCsc(OpKernelContext* context, const Index* posts, const Index* splits,
                        AdaptiveCscTable<Index>* table) {
  const GPUDevice& device = context->eigen_device<GPUDevice>();
  auto stream = device.stream();
  const auto type = DataTypeToEnum<Index>::v();
  const int64_t edges = table->edges;
  const int64_t rows = std::max<int64_t>(edges, 1);
  TF_RETURN_IF_ERROR(context->allocate_temp(type, TensorShape({table->n_post + 1}), &table->splits));
  TF_RETURN_IF_ERROR(context->allocate_temp(type, TensorShape({rows}), &table->edge_positions));
  TF_RETURN_IF_ERROR(context->allocate_temp(type, TensorShape({rows}), &table->sources));
  TF_RETURN_IF_ERROR(context->allocate_temp(type, TensorShape({rows}), &table->targets));
  Tensor iota;
  TF_RETURN_IF_ERROR(context->allocate_temp(type, TensorShape({rows}), &iota));
  const int blocks = static_cast<int>(std::min<int64_t>((rows + 255) / 256, 65535));
  if (edges > 0) {
    TF_RETURN_IF_ERROR(GpuLaunchKernel(CombinedIotaKernel<Index>, blocks, 256, 0, stream, edges,
                                       iota.flat<Index>().data()));
    size_t scratch_bytes = 0;
    if (cub::DeviceRadixSort::SortPairs(
            nullptr, scratch_bytes, posts, table->targets.template flat<Index>().data(),
            iota.flat<Index>().data(), table->edge_positions.template flat<Index>().data(), edges, 0,
            static_cast<int>(sizeof(Index) * 8), stream) != cudaSuccess) {
      cudaGetLastError();
      return errors::Internal("sizing the adaptive gather edge sort failed");
    }
    Tensor scratch;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_INT8, TensorShape({static_cast<int64_t>(scratch_bytes)}), &scratch));
    if (cub::DeviceRadixSort::SortPairs(
            scratch.flat<int8>().data(), scratch_bytes, posts, table->targets.template flat<Index>().data(),
            iota.flat<Index>().data(), table->edge_positions.template flat<Index>().data(), edges, 0,
            static_cast<int>(sizeof(Index) * 8), stream) != cudaSuccess) {
      cudaGetLastError();
      return errors::Internal("sorting adaptive gather edges by target failed");
    }
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        CombinedCscSourcesKernel<Index>, blocks, 256, 0, stream, edges, table->width, splits,
        table->edge_positions.template flat<Index>().data(), table->sources.template flat<Index>().data()));
  }
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      CombinedCscSplitsKernel<Index>, (table->n_post + 1 + 255) / 256, 256, 0, stream,
      table->n_post, edges, table->targets.template flat<Index>().data(),
      table->splits.template flat<Index>().data()));
  // The shared table must be complete before another operator can read it.
  const cudaError_t built = cudaStreamSynchronize(stream);
  if (built != cudaSuccess) {
    return errors::Internal("building adaptive gather tables failed: ", cudaGetErrorString(built));
  }
  const int64_t bytes = static_cast<int64_t>(table->splits.TotalBytes() +
                                             table->edge_positions.TotalBytes() +
                                             table->sources.TotalBytes() +
                                             table->targets.TotalBytes());
  DpointnetRoutingLedger::Get().Set(
      table, kRoutingCsc, {table->metadata.tensor_data().data(), kRoutingCsc, bytes, 1, 0});
  return OkStatus();
}

template <typename Index>
Status EnsureAdaptiveSegments(OpKernelContext* context, const Tensor& metadata, const Index* posts,
                              const Index* splits, int width, int64_t edges, int n_post, int tile,
                              bool uint8_types,
                              std::shared_ptr<AdaptiveSegmentTable<Index>>* held) {
  const DeviceBase* device = context->device();
  if (*held && AdaptiveTableMatches(**held, device, metadata, width, edges, n_post, uint8_types) &&
      (*held)->tile == tile) {
    return OkStatus();
  }
  auto& registry = GetAdaptiveRegistry<AdaptiveSegmentTable<Index>>();
  mutex_lock lock(registry.mu);
  for (auto it = registry.tables.begin(); it != registry.tables.end();) {
    auto table = it->lock();
    if (!table) {
      it = registry.tables.erase(it);
      continue;
    }
    if (AdaptiveTableMatches(*table, device, metadata, width, edges, n_post, uint8_types) &&
        table->tile == tile) {
      *held = std::move(table);
      return OkStatus();
    }
    ++it;
  }
  auto table = std::make_shared<AdaptiveSegmentTable<Index>>();
  table->device = device;
  table->metadata = metadata;
  table->metadata_elements = metadata.NumElements();
  table->width = width;
  table->edges = edges;
  table->n_post = n_post;
  table->tile = tile;
  table->uint8_types = uint8_types;
  TF_RETURN_IF_ERROR(BuildAdaptiveSegments<Index>(context, posts, splits, table.get()));
  registry.tables.emplace_back(table);
  *held = std::move(table);
  return OkStatus();
}

template <typename Index>
Status EnsureAdaptiveCsc(OpKernelContext* context, const Tensor& metadata, const Index* posts,
                         const Index* splits, int width, int64_t edges, int n_post,
                         bool uint8_types, std::shared_ptr<AdaptiveCscTable<Index>>* held) {
  const DeviceBase* device = context->device();
  if (*held && AdaptiveTableMatches(**held, device, metadata, width, edges, n_post, uint8_types)) {
    return OkStatus();
  }
  auto& registry = GetAdaptiveRegistry<AdaptiveCscTable<Index>>();
  mutex_lock lock(registry.mu);
  for (auto it = registry.tables.begin(); it != registry.tables.end();) {
    auto table = it->lock();
    if (!table) {
      it = registry.tables.erase(it);
      continue;
    }
    if (AdaptiveTableMatches(*table, device, metadata, width, edges, n_post, uint8_types)) {
      *held = std::move(table);
      return OkStatus();
    }
    ++it;
  }
  auto table = std::make_shared<AdaptiveCscTable<Index>>();
  table->device = device;
  table->metadata = metadata;
  table->metadata_elements = metadata.NumElements();
  table->width = width;
  table->edges = edges;
  table->n_post = n_post;
  table->uint8_types = uint8_types;
  TF_RETURN_IF_ERROR(BuildAdaptiveCsc<Index>(context, posts, splits, table.get()));
  registry.tables.emplace_back(table);
  *held = std::move(table);
  return OkStatus();
}

template <typename T>
struct AdaptiveIsActive {
  __host__ __device__ __forceinline__ bool operator()(const T& value) const {
    return static_cast<float>(value) > 0.0f;
  }
};

// queue[entry] = sample * width + row of each active slot in row-major order;
// starts[b] .. starts[b + 1] are sample b's entries. All on the device.
__global__ void AdaptiveSampleStartsKernel(unsigned int width, int batch,
                                           const unsigned int* queue,
                                           const unsigned int* queue_count,
                                           unsigned int* starts) {
  const unsigned int total = *queue_count;
  const unsigned int stride = gridDim.x * blockDim.x;
  const unsigned int first = blockIdx.x * blockDim.x + threadIdx.x;
  if (total == 0) {
    for (unsigned int sample = first; sample <= static_cast<unsigned int>(batch); sample += stride) {
      starts[sample] = 0;
    }
    return;
  }
  for (unsigned int index = first; index < total; index += stride) {
    const int sample = static_cast<int>(queue[index] / width);
    const int before = index == 0 ? -1 : static_cast<int>(queue[index - 1] / width);
    for (int covered = before + 1; covered <= sample; ++covered) starts[covered] = index;
    if (index == total - 1) {
      for (int covered = sample + 1; covered <= batch; ++covered) starts[covered] = total;
    }
  }
}

template <typename T>
Status BuildAdaptiveQueue(OpKernelContext* context, const T* spikes, int batch, int width,
                          unsigned int* queue, unsigned int* count, unsigned int* starts,
                          Tensor* scratch, int64_t* scratch_bytes_out) {
  auto stream = context->eigen_device<GPUDevice>().stream();
  const cub::CountingInputIterator<unsigned int> indices(0);
  const cub::TransformInputIterator<bool, AdaptiveIsActive<T>, const T*> active(
      spikes, AdaptiveIsActive<T>{});
  const int items = batch * width;
  size_t scratch_bytes = 0;
  if (cub::DeviceSelect::Flagged(nullptr, scratch_bytes, indices, active, queue, count, items,
                                 stream) != cudaSuccess) {
    cudaGetLastError();
    return errors::Internal("sizing the adaptive active-row queue failed");
  }
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(std::max<size_t>(scratch_bytes, 1))}), scratch));
  if (cub::DeviceSelect::Flagged(scratch->flat<int8>().data(), scratch_bytes, indices, active,
                                 queue, count, items, stream) != cudaSuccess) {
    cudaGetLastError();
    return errors::Internal("building the adaptive active-row queue failed");
  }
  *scratch_bytes_out = static_cast<int64_t>(scratch->TotalBytes());
  return GpuLaunchKernel(AdaptiveSampleStartsKernel, 64, 256, 0, stream,
                         static_cast<unsigned int>(width), batch, queue, count, starts);
}

template <typename T, typename Index>
struct AdaptiveSource {
  const T* spikes;
  const float* weights;
  const Index* posts;
  CsrTypeIds<Index> types;
  const Index* segment_splits;
  const Index* segment_starts;
  const unsigned int* segment_tiles;
  const unsigned long long* masks;
  const unsigned int* queue;
  const unsigned int* sample_starts;
  const Index* csc_splits;
  const Index* csc_edges;
  const Index* csc_sources;
  int width;
  int traversal;
};

template <typename T, typename Index>
struct AdaptiveSources {
  AdaptiveSource<T, Index> source[kCombinedSources];
};

// Row `row`'s segment in `tile`: [*first, *stop) in CSR order.
template <typename T, typename Index>
__device__ __forceinline__ bool AdaptiveFindSegment(const AdaptiveSource<T, Index>& source,
                                                    unsigned int row, unsigned int tile,
                                                    Index* first, Index* stop) {
  Index low = source.segment_splits[row];
  if (source.masks != nullptr) {
    const unsigned long long mask = source.masks[row];
    if (((mask >> tile) & 1ull) == 0) return false;
    low += static_cast<Index>(__popcll(mask & ((1ull << tile) - 1ull)));
  } else {
    const Index last = source.segment_splits[row + 1];
    Index high = last;
    while (low < high) {
      const Index middle = low + (high - low) / 2;
      if (source.segment_tiles[middle] < tile) low = middle + 1;
      else high = middle;
    }
    if (low >= last || source.segment_tiles[low] != tile) return false;
  }
  *first = source.segment_starts[low];
  *stop = source.segment_starts[low + 1];
  return true;
}

// Same arithmetic and equal-target run summation as the combined sweep's
// block walk; `segment` distinguishes rows so that only one row's run merges.
template <typename T, typename Index>
__device__ __forceinline__ void AdaptiveAccumulateRunEdge(
    const AdaptiveSource<T, Index>& source, const T* basis, Index edge, bool valid, float spike,
    int target_start, int lane, float* sums, unsigned int segment) {
  const unsigned int target = valid ? source.posts[edge] - target_start : 0xffffffffu;
  float contribution[4] = {};
  if (valid) {
    const float weighted = __fmul_rn(spike, source.weights[edge]);
    const int type = source.types[edge];
#pragma unroll
    for (int receptor = 0; receptor < 4; ++receptor) {
      contribution[receptor] = __fmul_rn(weighted, ToFloat(basis[type * 4 + receptor]));
    }
  }
  const unsigned int key = valid ? target | (segment << 13) : 0x80000000u | lane;
  const unsigned int below = __shfl_up_sync(0xffffffffu, key, 1);
  const bool head = lane == 0 || key != below;
  const unsigned int heads = __ballot_sync(0xffffffffu, head);
  if (heads != 0xffffffffu) {
    const unsigned int later = heads & ~((2u << lane) - 1);
    const int next = later ? __ffs(later) - 1 : 32;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const bool take = lane + offset < next;
      if (!__any_sync(0xffffffffu, take)) break;
#pragma unroll
      for (int receptor = 0; receptor < 4; ++receptor) {
        const float other = __shfl_down_sync(0xffffffffu, contribution[receptor], offset);
        if (take) contribution[receptor] += other;
      }
    }
  }
  if (valid && head) {
#pragma unroll
    for (int receptor = 0; receptor < 4; ++receptor) {
      atomicAdd(sums + target * 4 + receptor, contribution[receptor]);
    }
  }
}

// Ticketed warp batches for target-unique rows. Each warp claims `rows`
// consecutive active rows from the block's ticket, finds their segments in
// this tile, compacts the found ones into its own 32-entry window and walks
// their concatenated edges, kEdges per lane per step with all loads issued
// before the shared atomics. The batch size adapts to the sample's activity:
// about eight batches per warp, 4 to 32 rows.
template <int kEdges, typename T, typename Index>
__device__ __forceinline__ void AdaptiveWarpBatches(
    const AdaptiveSource<T, Index>& source, int sample, unsigned int tile, int target_start,
    const T* basis, float* sums, Index* starts, Index* offsets, float* amplitudes,
    unsigned int* ticket) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  Index* my_starts = starts + warp * 32;
  Index* my_offsets = offsets + warp * 32;
  float* my_spikes = amplitudes + warp * 32;
  const unsigned int begin = source.sample_starts[sample];
  const unsigned int count = source.sample_starts[sample + 1] - begin;
  const unsigned int share = count / (static_cast<unsigned int>(warps) * kAdaptiveBatchShare);
  const unsigned int rows = share >= 32 ? 32 : (share >= 16 ? 16 : (share >= 8 ? 8 : 4));
  const unsigned int sample_first = static_cast<unsigned int>(sample) * source.width;
  for (;;) {
    unsigned int base = 0;
    if (lane == 0) base = atomicAdd(ticket, rows);
    base = __shfl_sync(0xffffffffu, base, 0);
    if (base >= count) break;
    const unsigned int index = base + lane;
    Index first = 0, stop = 0;
    float spike = 0.0f;
    bool found = false;
    if (lane < rows && index < count) {
      const unsigned int row = source.queue[begin + index] - sample_first;
      found = AdaptiveFindSegment(source, row, tile, &first, &stop);
      if (found) spike = ToFloat(source.spikes[static_cast<int64_t>(sample) * source.width + row]);
    }
    const Index length = found ? stop - first : Index(0);
    Index prefix = length;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const Index other = __shfl_up_sync(0xffffffffu, prefix, offset);
      if (lane >= offset) prefix += other;
    }
    const Index total = __shfl_sync(0xffffffffu, prefix, 31);
    const unsigned int found_lanes = __ballot_sync(0xffffffffu, found);
    const int items = __popc(found_lanes);
    if (found) {
      const int item = __popc(found_lanes & ((1u << lane) - 1u));
      my_starts[item] = first;
      my_offsets[item] = prefix - length;
      my_spikes[item] = spike;
    }
    __syncwarp();
    int owner = 0;
    for (Index chunk = 0; chunk < total; chunk += 32 * kEdges) {
      bool valid[kEdges];
      Index edge[kEdges];
      float amplitude[kEdges];
#pragma unroll
      for (int step = 0; step < kEdges; ++step) {
        const Index position = chunk + step * 32 + lane;
        valid[step] = position < total;
        edge[step] = 0;
        amplitude[step] = 0.0f;
        if (valid[step]) {
          while (owner + 1 < items && my_offsets[owner + 1] <= position) ++owner;
          edge[step] = my_starts[owner] + (position - my_offsets[owner]);
          amplitude[step] = my_spikes[owner];
        }
      }
      float weight[kEdges];
      unsigned int target[kEdges];
      int type[kEdges];
#pragma unroll
      for (int step = 0; step < kEdges; ++step) {
        weight[step] = valid[step] ? source.weights[edge[step]] : 0.0f;
        target[step] = valid[step] ? static_cast<unsigned int>(source.posts[edge[step]] - target_start) : 0u;
        type[step] = valid[step] ? static_cast<int>(source.types[edge[step]]) : 0;
      }
#pragma unroll
      for (int step = 0; step < kEdges; ++step) {
        if (!valid[step]) continue;
        const float weighted = __fmul_rn(amplitude[step], weight[step]);
#pragma unroll
        for (int receptor = 0; receptor < 4; ++receptor) {
          atomicAdd(sums + target[step] * 4 + receptor,
                    __fmul_rn(weighted, ToFloat(basis[type[step] * 4 + receptor])));
        }
      }
    }
    __syncwarp();
  }
}

// Block rounds for target-repeating or high-fanout rows: one thread per active
// row finds its segment, one packed block scan yields each found row's slot
// and edge prefix, and warps walk 32-edge chunks of the concatenated segments
// (warp-uniform owner search, then a per-lane linear advance).
template <int kThreads, typename T, typename Index, typename PackedScan>
__device__ __forceinline__ void AdaptiveBlockRounds(
    const AdaptiveSource<T, Index>& source, int sample, unsigned int tile, int target_start,
    const T* basis, float* sums, Index* starts, Index* offsets, float* amplitudes,
    typename PackedScan::TempStorage& scan_storage) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const unsigned int begin = source.sample_starts[sample];
  const unsigned int count = source.sample_starts[sample + 1] - begin;
  const unsigned int sample_first = static_cast<unsigned int>(sample) * source.width;
  constexpr int kCountBits = kThreads == 1024 ? 11 : 10;
  constexpr int kCountShift = 64 - kCountBits;
  for (unsigned int round = 0; round < count; round += kThreads) {
    const unsigned int item = round + threadIdx.x;
    Index first = 0, stop = 0;
    float spike = 0.0f;
    bool found = false;
    if (item < count) {
      const unsigned int row = source.queue[begin + item] - sample_first;
      found = AdaptiveFindSegment(source, row, tile, &first, &stop);
      if (found) spike = ToFloat(source.spikes[static_cast<int64_t>(sample) * source.width + row]);
    }
    const unsigned long long length = found ? static_cast<unsigned long long>(stop - first) : 0ull;
    const unsigned long long packed = length ? (1ull << kCountShift) | length : 0ull;
    unsigned long long prefix, aggregate;
    PackedScan(scan_storage).ExclusiveSum(packed, prefix, aggregate);
    const int found_rows = static_cast<int>(aggregate >> kCountShift);
    const Index total = static_cast<Index>(aggregate & ((1ull << kCountShift) - 1));
    if (length) {
      const int slot = static_cast<int>(prefix >> kCountShift);
      starts[slot] = first;
      offsets[slot] = static_cast<Index>(prefix & ((1ull << kCountShift) - 1));
      amplitudes[slot] = spike;
    }
    if (threadIdx.x == 0) offsets[found_rows] = total;
    __syncthreads();
    for (Index chunk = warp * 32; chunk < total; chunk += kThreads) {
      int owner = 0, high = found_rows;
      while (high - owner > 1) {
        const int middle = (owner + high) / 2;
        if (offsets[middle] <= chunk) owner = middle;
        else high = middle;
      }
      const Index position = chunk + lane;
      const bool valid = position < total;
      if (valid) {
        while (offsets[owner + 1] <= position) ++owner;
      }
      const Index edge = valid ? starts[owner] + (position - offsets[owner]) : Index(0);
      AdaptiveAccumulateRunEdge(source, basis, edge, valid, amplitudes[owner], target_start, lane,
                                sums, static_cast<unsigned int>(owner));
    }
    __syncthreads();
  }
}

// Kept in a nested namespace under the combined-sweep kernel name so profiler
// kernel names contain "DpointnetCombinedSweepKernel" in both schemas, while
// "adaptive_routing::" and the AdaptiveSources parameter distinguish it.
namespace adaptive_routing {

template <typename T, typename Index, int kTargets, int kThreads>
__global__ void __launch_bounds__(kThreads, 1024 / kThreads) DpointnetCombinedSweepKernel(
    AdaptiveSources<T, Index> sources, int source_count, int n_post, const T* basis,
    bool batch_fast, T* output) {
  static_assert(kThreads == 512 || kThreads == 1024, "adaptive sweep block size");
  __shared__ float static_sums[kTargets <= 2048 ? kTargets * 4 : 1];
  extern __shared__ float dynamic_sums[];
  float* sums = kTargets <= 2048 ? static_sums : dynamic_sums;
  using PackedScan = cub::BlockScan<unsigned long long, kThreads, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename PackedScan::TempStorage scan_storage;
  __shared__ Index starts[kThreads];
  __shared__ Index offsets[kThreads + 1];
  __shared__ float amplitudes[kThreads];
  __shared__ unsigned int ticket;
  const int tile = batch_fast ? blockIdx.y : blockIdx.x;
  const int target_start = tile * kTargets;
  const int sample = batch_fast ? blockIdx.x : blockIdx.y;
  for (int item = threadIdx.x; item < kTargets * 4; item += kThreads) sums[item] = 0.0f;
  for (int source_id = 0; source_id < source_count; ++source_id) {
    const auto source = sources.source[source_id];
    if (source.traversal == kAdaptiveGather || source.queue == nullptr) continue;
    if (threadIdx.x == 0) ticket = 0;
    __syncthreads();
    if (source.traversal == kAdaptiveWarpBatches) {
      AdaptiveWarpBatches<kAdaptiveRecurrentEdges>(source, sample, static_cast<unsigned int>(tile), target_start, basis,
                             sums, starts, offsets, amplitudes, &ticket);
    } else {
      AdaptiveBlockRounds<kThreads, T, Index, PackedScan>(
          source, sample, static_cast<unsigned int>(tile), target_start, basis, sums, starts,
          offsets, amplitudes, scan_storage);
    }
    __syncthreads();
  }
  __syncthreads();
  const int targets = min(kTargets, n_post - target_start);
  for (int source_id = 0; source_id < source_count; ++source_id) {
    const auto source = sources.source[source_id];
    if (source.traversal != kAdaptiveGather || source.csc_splits == nullptr) continue;
    const T* activity = source.spikes + static_cast<int64_t>(sample) * source.width;
    constexpr int kGroup = 4;
#pragma unroll 2
    for (int item = threadIdx.x; item < targets; item += kThreads) {
      const Index begin = source.csc_splits[target_start + item];
      const Index end = source.csc_splits[target_start + item + 1];
      float acc[4] = {};
      for (Index base = begin; base < end; base += kGroup) {
        float spike[kGroup];
        Index edge[kGroup];
#pragma unroll
        for (int k = 0; k < kGroup; ++k) {
          const bool valid = base + k < end;
          spike[k] = valid ? ToFloat(activity[source.csc_sources[base + k]]) : 0.0f;
        }
#pragma unroll
        for (int k = 0; k < kGroup; ++k) {
          edge[k] = spike[k] > 0.0f ? source.csc_edges[base + k] : 0;
        }
        float weight[kGroup];
        int type[kGroup];
#pragma unroll
        for (int k = 0; k < kGroup; ++k) {
          weight[k] = spike[k] > 0.0f ? source.weights[edge[k]] : 0.0f;
          type[k] = spike[k] > 0.0f ? static_cast<int>(source.types[edge[k]]) : 0;
        }
#pragma unroll
        for (int k = 0; k < kGroup; ++k) {
          if (spike[k] > 0.0f) {
            const float weighted = __fmul_rn(spike[k], weight[k]);
#pragma unroll
            for (int receptor = 0; receptor < 4; ++receptor) {
              acc[receptor] += __fmul_rn(weighted, ToFloat(basis[type[k] * 4 + receptor]));
            }
          }
        }
      }
#pragma unroll
      for (int receptor = 0; receptor < 4; ++receptor) {
        sums[item * 4 + receptor] += acc[receptor];
      }
    }
  }
  __syncthreads();
  for (int item = threadIdx.x; item < targets * 4; item += kThreads) {
    output[(static_cast<int64_t>(sample) * n_post + target_start) * 4 + item] =
        FromFloat<T>(sums[item]);
  }
}

}  // namespace adaptive_routing

template <typename T, typename Index, int kTargets>
struct AdaptiveSweepShape {
  static constexpr int kThreads = kTargets > 2048 && sizeof(Index) == 4 ? 1024 : 512;
  static constexpr int kSharedBytes = kTargets > 2048 ? kTargets * 4 * sizeof(float) : 0;
  static const void* Kernel() {
    return reinterpret_cast<const void*>(adaptive_routing::DpointnetCombinedSweepKernel<T, Index, kTargets, kThreads>);
  }
};

// Calls launch(std::integral_constant<int, Tile>) for a supported tile width.
template <typename Launch>
Status WithAdaptiveTile(int tile, Launch&& launch) {
  switch (tile) {
    case 256: return launch(std::integral_constant<int, 256>{});
    case 512: return launch(std::integral_constant<int, 512>{});
    case 1024: return launch(std::integral_constant<int, 1024>{});
    case 2048: return launch(std::integral_constant<int, 2048>{});
    case 3072: return launch(std::integral_constant<int, 3072>{});
    case 4096: return launch(std::integral_constant<int, 4096>{});
    case 5120: return launch(std::integral_constant<int, 5120>{});
  }
  return errors::InvalidArgument("Unsupported adaptive routing tile width ", tile, ".");
}
