#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/lib/core/errors.h>

using namespace tensorflow;
template<typename T> using EigenTensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;

template<typename IndexType>
class PackedToSparseIndexOp : public OpKernel {
public:
    explicit PackedToSparseIndexOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* lengths;

        OP_REQUIRES_OK(ctx, ctx->input("lengths", &lengths));
        auto lengths_vec = lengths->vec<IndexType>();

        EigenTensorScalar<IndexType> num_elements = lengths_vec.sum();

        Tensor* output;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("indices", {num_elements(), 2}, &output));

        auto output_mat = output->matrix<IndexType>();

        IndexType current_offset = 0;

        for(IndexType i = 0; i < lengths_vec.dimension(0); ++i) {
            auto length_i = lengths_vec(i);

            for(IndexType j = 0; j < length_i; ++j) {
                output_mat(current_offset, 0) = i;
                output_mat(current_offset, 1) = j;
                ++current_offset;
            }
        }
    }
};

REGISTER_OP("PackedToSparseIndex")
    .Attr("IndexT: {int32, int64} = DT_INT32")
    .Input("lengths: IndexT")
    .Output("indices: IndexT")
    .SetShapeFn([](shape_inference::InferenceContext* c){
        shape_inference::ShapeHandle input_shape;

        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));
        c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 2));

        return Status::OK();
    }).Doc(
R"doc(
Converts a vector describing lengths of a ragged array into indices into a matrix
which are compatible with SparseTensor.

Parameters
----------
lengths: a 1-dimensional array of integers representing the lengths.

Returns
-------
indices: a 2-dimensional tensor representing indices of the ragged array.
)doc");

#define REGISTER(T) REGISTER_KERNEL_BUILDER(\
    Name("PackedToSparseIndex") \
    .Device("CPU") \
    .TypeConstraint<T>("IndexT"), \
    PackedToSparseIndexOp<T>);

REGISTER(int32)
REGISTER(int64)

#undef REGISTER