#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/util/work_sharder.h>

using namespace tensorflow;

template<typename T> using EigenTensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;

/* Operation to compute segments in a padded batch of packed lengths.
 *
 * This operation attempts to perform the following: given an array of lengths
 * [a b c d ...] it transforms that to the corresponding segment description
 * which is valid for `segment_sum` and similar, where each segment is contiguous
 * and has the provided description.
 * 
 * This operation generalizes the above transformation in the case where we have
 * a batch of such lengths, which are possibly padded at the end. If there is some
 * padding, a segment is generated for the padded indices, and then the corresponding
 * number of zero length sequences are generated to pad out the array.
 * 
 */
template<typename IndexType>
class BatchLengthToSegmentOp : public OpKernel {
public:
    explicit BatchLengthToSegmentOp(OpKernelConstruction* ctx): OpKernel(ctx) {
    }

    void Compute(OpKernelContext* ctx) override {
        typedef Eigen::array<int64, 2> SliceIndexType;

        auto lengths = ctx->input(0);
        auto num_cols_tensor = ctx->input(1);

        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(lengths.shape()),
                    errors::InvalidArgument("lengths must be a matrix."));
        

        int64 num_rows = lengths.dim_size(0);
        int64 num_input_cols = lengths.dim_size(1);
        int64 num_output_cols = num_cols_tensor.scalar<IndexType>()();

        OP_REQUIRES(ctx, num_output_cols > 0,
                    errors::InvalidArgument("number of output columns must be larger than 0."));

        auto lengths_mat = lengths.matrix<IndexType>();

        Tensor* output;

        OP_REQUIRES_OK(ctx, ctx->allocate_output("segment", {num_rows, num_output_cols}, &output));
        auto output_mat = output->matrix<IndexType>();

        int64 num_segments_per_row = num_input_cols + 1;

        auto work = [&](int64 start, int64 end) {
            for(int64 i = start; i < end; ++i) {
                int64 current_offset = 0;
                int64 j = 0;

                while (j < num_input_cols) {
                    auto segment_length = lengths_mat(i, j);

                    if (segment_length == 0) {
                        break;
                    }

                    std::fill(
                        &output_mat(i, current_offset),
                        &output_mat(i, current_offset) + segment_length,
                        i * num_segments_per_row + j);

                    current_offset += segment_length;
                    ++j;
                }

                if (current_offset != num_output_cols) {
                    // this row was padded.
                    SliceIndexType start = {i, current_offset};
                    SliceIndexType size = {1, num_output_cols - current_offset};

                    output_mat.slice(start, size).setConstant((i + 1) * num_segments_per_row - 1);
                }
            }
        };

        auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
        Shard(worker_threads->num_threads, worker_threads->workers, num_rows,
              5 * num_output_cols, work);
    }
};


REGISTER_OP("BatchLengthToSegment")
    .Attr("IndexT: {int32, int64} = DT_INT32")
    .Input("lengths: IndexT")
    .Input("output_columns: IndexT")
    .Output("segment: IndexT")
    .SetShapeFn([](shape_inference::InferenceContext* c){
        shape_inference::ShapeHandle input_shape;

        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

        shape_inference::DimensionHandle num_output_cols;
        c->MakeDimForScalarInput(0, &num_output_cols);

        c->set_output(0, c->Matrix(c->Dim(input_shape, 0), num_output_cols));

        return Status::OK();
    }).Doc(
R"doc(
Computes segments from a padded batch.

Parameters
----------
lengths: a matrix of segment lengths, potentially zero padded.
output_columns: the total number of items per row in the output.

Returns
-------
segment: a matrix representing the segment for each row.
)doc");

#define REGISTER(T) REGISTER_KERNEL_BUILDER( \
    Name("BatchLengthToSegment") \
    .Device("CPU") \
    .TypeConstraint<T>("IndexT"), \
    BatchLengthToSegmentOp<T>)

REGISTER(int32)
REGISTER(int64)

#undef REGISTER