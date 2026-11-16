// Host record store for the direct recorded adjoint: exact, asynchronous
// offload of per-step records (FP32 pre-reset membrane rows, packed spike/
// refractory word frames) to pinned host memory during the forward loop, and
// prefetched return of the same bytes during the reverse loop.
//
// Design adapted from Javier Galvan's v1_model_utils/cuda_frame_store raw mode
// (commit 5d58bc3; credit: Javier). DPointNet keeps the store codec-free (no
// compression), copies frames from a small device staging ring the forward
// kernels already write in place, and hands prefetched frames back as ordinary
// device tensors; it never changes the arithmetic of the recorded adjoint.
//
// The store owns one pinned host buffer of `frames * prod(frame_shape)`
// elements, a dedicated CUDA copy stream and one completion event per frame.
// All device work is stream-ordered: offloads wait for the compute stream to
// reach the staging write before they read, the compute stream waits for a
// frame's offload before the forward may reuse its staging slot, and takes
// wait for the frame's host-to-device copy before the output is consumed.
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("DpointnetHostRecordStore")
    .Attr("frames: int >= 1")
    .Attr("frame_shape: shape")
    .Attr("dtype: {half, float}")
    .Attr("prefetch: int >= 1 = 8")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("resource: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
A pinned-host store of `frames` equally shaped records, created on first use.
`prefetch` bounds the number of frames kept resident on the device after a take.
)doc");

REGISTER_OP("DpointnetHostRecordOffload")
    .Attr("T: {half, float}")
    .Attr("A: type")
    .Attr("staging_slots: int >= 1")
    .Input("store: resource")
    .Input("staging: resource")
    .Input("slot: int32")
    .Input("index: int32")
    .Input("after: A")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Copies frame `slot` of the `[staging_slots, *frame_shape]` resource variable
`staging` into host frame `index` on the store's copy stream, after the compute
stream has produced `after`. The compute stream is then made to wait for the
offload of frame `index + 1 - staging_slots`, whose staging slot the next
forward step overwrites.
)doc");

REGISTER_OP("DpointnetHostRecordTake")
    .Attr("T: {half, float}")
    .Input("store: resource")
    .Input("index: int32")
    .Input("prefetch: int32")
    .Output("frame: T")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Returns host frame `index` as a device tensor, reusing its prefetched copy when
present, and starts the host-to-device copy of frame `prefetch` (ignored when
negative or beyond the store) on the copy stream.
)doc");

REGISTER_OP("DpointnetHostRecordPrefetch")
    .Attr("A: type")
    .Input("store: resource")
    .Input("index: int32")
    .Input("after: A")
    .Output("token: int32")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Starts the host-to-device copy of frame `index` after `after` is produced, so a
later take finds it resident. Out-of-range indices are ignored.
)doc");

REGISTER_OP("DpointnetHostRecordStats")
    .Input("store: resource")
    .Output("stats: int64")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      context->set_output(0, context->Vector(6));
      return absl::OkStatus();
    })
    .Doc(R"doc(
`[frames, frame_bytes, host_bytes, offloaded_frames, resident_frames, takes]`.
)doc");
