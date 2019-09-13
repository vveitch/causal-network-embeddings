#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/lib/core/errors.h>

using namespace tensorflow;

template<typename T> using EigenTensorScalar = Eigen::Tensor<T, 0, Eigen::RowMajor>;

template<typename IndexType>
class GetOpenEgoNetworkOp : public OpKernel {
public:
    explicit GetOpenEgoNetworkOp(OpKernelConstruction* ctx): OpKernel(ctx) {
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* neighbours;
        const Tensor* lengths;
        const Tensor* offsets;
        const Tensor* centers;

        OP_REQUIRES_OK(ctx, ctx->input("centers", &centers));
        OP_REQUIRES_OK(ctx, ctx->input("neighbours", &neighbours));
        OP_REQUIRES_OK(ctx, ctx->input("lengths", &lengths));
        OP_REQUIRES_OK(ctx, ctx->input("offsets", &offsets));

        auto centers_vec = centers->vec<IndexType>();
        auto neighbours_vec = neighbours->vec<IndexType>();
        auto lengths_vec = lengths->vec<IndexType>();
        auto offsets_vec = offsets->vec<IndexType>();

        EigenTensorScalar<IndexType> total_neighbours;
        total_neighbours = centers_vec.unaryExpr([&](IndexType x) { return lengths_vec(x); }).sum();

        Tensor* edge_list;

        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {total_neighbours(), 2}, &edge_list));
        auto edge_list_vec = edge_list->matrix<IndexType>();

        IndexType current_offset = 0;

        for(IndexType i = 0; i < centers_vec.dimension(0); ++i) {
            typedef Eigen::array<IndexType, 1> NeighboursSliceIndex;
            typedef Eigen::array<IndexType, 2> EdgeSliceIndex;

            auto offset_i = offsets_vec(centers_vec(i));
            auto length_i = lengths_vec(centers_vec(i));

            NeighboursSliceIndex n_slice_offsets =  {offset_i};
            NeighboursSliceIndex n_slice_lengths = {length_i};

            auto neighbours_i = neighbours_vec.slice(n_slice_offsets, n_slice_lengths);

            EdgeSliceIndex e_slice_offsets_start = {current_offset, 0};
            EdgeSliceIndex e_slice_offsets_end = {current_offset, 1};
            EdgeSliceIndex e_slice_lengths = {length_i, 1};

            edge_list_vec.slice(e_slice_offsets_start, e_slice_lengths) = neighbours_i.constant(i);
            edge_list_vec.slice(e_slice_offsets_end, e_slice_lengths) = neighbours_i;
            current_offset += length_i;
        }
    }
};


REGISTER_OP("GetOpenEgoNetwork")
    .Attr("T: {int32, int64} = DT_INT32")
    .Input("centers: T")
    .Input("neighbours: T")
    .Input("lengths: T")
    .Input("offsets: T")
    .Output("edge_list: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape;

        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &input_shape));

        c->set_output(0, c->Matrix(-1, 2));

        return Status::OK();
    });

#define REGISTER(TYPE) \
    REGISTER_KERNEL_BUILDER(Name("GetOpenEgoNetwork")\
        .Device("CPU")\
        .TypeConstraint<TYPE>("T"),\
        GetOpenEgoNetworkOp<TYPE>)

REGISTER(int32)
REGISTER(int64)

#undef REGISTER