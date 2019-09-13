#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/lib/core/errors.h>

using namespace tensorflow;

template<typename T> using EigenTensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;

template<typename SourceType, typename IndexType>
class ConcatenateSlicesOp : public OpKernel {
public:
    explicit ConcatenateSlicesOp(OpKernelConstruction* ctx): OpKernel(ctx) {
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* source;
        const Tensor* begins;
        const Tensor* sizes;
        Tensor* output;

        OP_REQUIRES_OK(ctx, ctx->input("source", &source));
        OP_REQUIRES_OK(ctx, ctx->input("begins", &begins));
        OP_REQUIRES_OK(ctx, ctx->input("sizes", &sizes));

        auto sources_vec = source->vec<SourceType>();
        auto begins_vec = begins->vec<IndexType>();
        auto sizes_vec = sizes->vec<IndexType>();

        EigenTensorScalar<IndexType> num_elements = sizes_vec.sum();

        OP_REQUIRES_OK(ctx, ctx->allocate_output("concat_slices", {num_elements()}, &output));

        auto output_vec = output->vec<SourceType>();
        auto num_slices = begins_vec.dimension(0);

        OP_REQUIRES(ctx, num_slices == sizes_vec.dimension(0),
                    errors::InvalidArgument("begins and sizes must have the same length"));

        IndexType current_offset = 0;

        for(int64 i = 0; i < num_slices; ++i) {
            typedef Eigen::array<IndexType, 1> SliceIndexType;

            auto slice_length = sizes_vec(i);

            SliceIndexType current_offset_s = {current_offset};
            SliceIndexType slice_start_s = {begins_vec(i)};
            SliceIndexType slice_length_s = {slice_length};

            output_vec.slice(current_offset_s, slice_length_s) = sources_vec.slice(slice_start_s, slice_length_s);
            current_offset += slice_length;
        }
    }
};

REGISTER_OP("ConcatenateSlices")
    .Attr("SourceT: numbertype")
    .Attr("IndexT: {int32, int64} = DT_INT32")
    .Input("source: SourceT")
    .Input("begins: IndexT")
    .Input("sizes: IndexT")
    .Output("concat_slices: SourceT")
    .SetShapeFn([](shape_inference::InferenceContext* c){
        shape_inference::ShapeHandle input_shape;

        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input_shape));

        shape_inference::DimensionHandle input_dim;
        TF_RETURN_IF_ERROR(c->Merge(
            c->Dim(c->input(1), 0),
            c->Dim(c->input(2), 0),
            &input_dim));

        c->set_output(0, c->Vector(-1));

        return Status::OK();
    }).Doc(
R"doc(
Concatenates slices from a given tensor.

Parameters
----------
source: the tensor from which to extract the slices.
begins: a 1-dimensional tensor representing the initial indices of the slices.
sizes: a 1-dimensional tensor representing the lengths of each slice.

Returns
-------
concat_slices: a tensor with the extracted slices concatenated.
)doc");


#define REGISTER(T, U) REGISTER_KERNEL_BUILDER(\
    Name("ConcatenateSlices") \
    .Device("CPU") \
    .TypeConstraint<T>("SourceT") \
    .TypeConstraint<U>("IndexT"), \
    ConcatenateSlicesOp<T, U>);

#define REGISTER_INTS(T) REGISTER(T, int32) REGISTER(T, int64)

TF_CALL_NUMBER_TYPES(REGISTER_INTS)

#undef REGISTER_INTS
#undef REGISTER