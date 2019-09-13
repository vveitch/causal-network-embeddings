#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/lib/core/errors.h>

using namespace tensorflow;

template<typename T> using EigenTensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;


template<typename ValueType, typename IndexType>
void RepeatScalar(const Tensor& values, const Tensor& counts, OpKernelContext* ctx) {
    typedef Eigen::array<int64, 1> SliceIndexType;

    Tensor* output;

    int64 values_length = values.dim_size(0);
    int64 repeat_count = counts.scalar<IndexType>()();
    int64 output_length = values_length * repeat_count;

    OP_REQUIRES_OK(ctx, ctx->allocate_output("output", {output_length}, &output));

    auto values_vec = values.vec<ValueType>();
    auto output_vec = output->vec<ValueType>();

    for(int64 i = 0; i < values_length; ++i) {
        SliceIndexType start = {i * repeat_count};
        SliceIndexType size = {repeat_count};

        output_vec.slice(start, size).setConstant(values_vec(i));
    }
}

template<typename ValueType, typename IndexType>
void RepeatVector(const Tensor&  values, const Tensor& counts, OpKernelContext* ctx) {
    typedef Eigen::array<int64, 1> SliceIndexType;

    Tensor* output;

    auto counts_vec = counts.vec<IndexType>();
    EigenTensorScalar<IndexType> output_length = counts_vec.sum();

    OP_REQUIRES_OK(ctx, ctx->allocate_output("output", {output_length()}, &output));

    auto values_vec = values.vec<ValueType>();
    auto output_vec = output->vec<ValueType>();

    int64 values_length = values_vec.dimension(0);
    int64 current_offset = 0;

    for(int64 i = 0; i < values_length; ++i) {
        SliceIndexType start = {current_offset};
        SliceIndexType size = {counts_vec(i)};

        output_vec.slice(start, size).setConstant(values_vec(i));
        current_offset += counts_vec(i);
    }
}


template<typename ValueType, typename IndexType>
class RepeatOp : public OpKernel {
public:
    explicit RepeatOp(OpKernelConstruction* ctx): OpKernel(ctx) {
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* values;
        const Tensor* counts;

        OP_REQUIRES_OK(ctx, ctx->input("values", &values));
        OP_REQUIRES_OK(ctx, ctx->input("counts", &counts));

        OP_REQUIRES(ctx, TensorShapeUtils::IsVector(values->shape()),
                    errors::InvalidArgument("values should be vector."));
        
        OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(counts->shape()) || TensorShapeUtils::IsVector(counts->shape()),
                    errors::InvalidArgument("counts should be scalar or vector."));
        
        if (TensorShapeUtils::IsScalar(counts->shape())) {
            RepeatScalar<ValueType, IndexType>(*values, *counts, ctx);
        } else {
            OP_REQUIRES(ctx, values->dim_size(0) == counts->dim_size(0),
                        errors::InvalidArgument("If counts is a vector, it must have the same length as values."));
            RepeatVector<ValueType, IndexType>(*values, *counts, ctx);
        }
    }
};

REGISTER_OP("Repeat")
    .Attr("ValueT: numbertype")
    .Attr("IndexT: {int32, int64} = DT_INT32")
    .Input("values: ValueT")
    .Input("counts: IndexT")
    .Output("output: ValueT")
    .SetShapeFn([](shape_inference::InferenceContext* c){
        shape_inference::ShapeHandle input_shape;

        TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &input_shape));

        if(c->Rank(input_shape) == 0) {
            shape_inference::DimensionHandle scalar_count;
            TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(0, &scalar_count));
            TF_RETURN_IF_ERROR(c->Multiply(scalar_count, c->Dim(input_shape, 0), &scalar_count));
            c->set_output(0, c->Vector(scalar_count));
        } else {
            TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &input_shape));
            c->set_output(0, c->Vector(-1));
        }

        return Status::OK();
    }).Doc(
R"doc(
Repeats the given vector by a fixed number of counts.

Parameters
----------
values: the values to repeat.
counts: either a scalar, to repeat each value a fixed number of times, or a vector of the same length as
    values, to repeat each value that many times.

Returns
-------
output: an output vector containing the repeated values.
)doc");

#define REGISTER(T, U) REGISTER_KERNEL_BUILDER(\
    Name("Repeat") \
    .Device("CPU") \
    .TypeConstraint<T>("ValueT") \
    .TypeConstraint<U>("IndexT"), \
    RepeatOp<T, U>);


#define REGISTER_INTS(T) REGISTER(T, int32) REGISTER(T, int64)
TF_CALL_NUMBER_TYPES(REGISTER_INTS)

#undef REGISTER_INTS
#undef REGISTER
