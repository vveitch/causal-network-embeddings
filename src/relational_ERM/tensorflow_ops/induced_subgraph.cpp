#include <algorithm>
#include <vector>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/lib/core/errors.h>

using namespace tensorflow;


class GetInducedSubgraphOp : public OpKernel {
public:
    explicit GetInducedSubgraphOp(OpKernelConstruction* ctx): OpKernel(ctx) {
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* vertex;
        const Tensor* neighbours;
        const Tensor* offsets;
        const Tensor* lengths;

        // this tensor is used to store the inverse map
        // from global vertex index to local vertex index.
        Tensor temporary_index_map;

        Tensor* output_lengths;
        Tensor* output_neighbours;
        Tensor* output_offsets;

        OP_REQUIRES_OK(ctx, ctx->input("vertex", &vertex));
        OP_REQUIRES_OK(ctx, ctx->input("neighbours", &neighbours));
        OP_REQUIRES_OK(ctx, ctx->input("offsets", &offsets));
        OP_REQUIRES_OK(ctx, ctx->input("lengths", &lengths));

        auto const& vertex_vec = vertex->vec<int32>();
        auto const& neighbours_vec = neighbours->vec<int32>();
        auto const& offsets_vec = offsets->vec<int32>();
        auto const& lengths_vec = lengths->vec<int32>();

        auto num_subgraph_vertex = vertex_vec.dimension(0);

        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, lengths->shape(), &temporary_index_map));

        auto temp_index_map_vec = temporary_index_map.vec<int32>();
        temp_index_map_vec.setConstant(-1);

        for(int i = 0; i < num_subgraph_vertex; ++i) {
            temp_index_map_vec(vertex_vec(i)) = i;
        }

        OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({"vertex"}, {"subgraph_offsets"}, {num_subgraph_vertex}, &output_offsets));
        OP_REQUIRES_OK(ctx, ctx->allocate_output("subgraph_lengths", {num_subgraph_vertex}, &output_lengths));

        auto output_offsets_vec = output_offsets->vec<int32>();
        auto output_lengths_vec = output_lengths->vec<int32>();

        int num_subgraph_edges = 0;

        for(int i = 0; i < num_subgraph_vertex; ++i) {
            Eigen::array<int32, 1> num_neighbours_i = {lengths_vec(vertex_vec(i))};
            Eigen::array<int32, 1> offset_i = {offsets_vec(vertex_vec(i))};

            auto neighbours_new_idx = neighbours_vec
                .slice(offset_i, num_neighbours_i)
                .unaryExpr([&](int32 v) { return temp_index_map_vec(v); });

            Eigen::Tensor<int32, 0, Eigen::RowMajor> num_subgraph_neighbours = (neighbours_new_idx != -1).cast<int32>().sum();
            num_subgraph_edges += num_subgraph_neighbours();
            output_lengths_vec(i) = num_subgraph_neighbours();
        }

        OP_REQUIRES_OK(ctx, ctx->allocate_output("subgraph_neighbours", {num_subgraph_edges}, &output_neighbours));
        auto output_neighbours_vec = output_neighbours->vec<int32>();

        int current_offset = 0;

        for(int i = 0; i < num_subgraph_vertex; ++i) {
            auto num_neighbours_i = lengths_vec(vertex_vec(i));
            auto offset_source = offsets_vec(vertex_vec(i));
            output_offsets_vec(i) = current_offset;

            for(int j = 0; j < num_neighbours_i; ++j) {
                auto n = neighbours_vec(offset_source + j);
                auto new_n = temp_index_map_vec(n);

                if (new_n == -1) {
                    // skip vertex not in induced subgraph.
                    continue;
                }

                output_neighbours_vec(current_offset) = new_n;
                ++current_offset;
            }
        }
    }
};

REGISTER_OP("GetInducedSubgraph")
    .Input("vertex: int32")
    .Input("neighbours: int32")
    .Input("lengths: int32")
    .Input("offsets: int32")
    .Output("subgraph_neighbours: int32")
    .Output("subgraph_lengths: int32")
    .Output("subgraph_offsets: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape;

        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &input_shape));

        auto num_vertex = c->Dim(c->input(0), 0);

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(num_vertex));
        c->set_output(2, c->Vector(num_vertex));

        return Status::OK();
    }).Doc(
R"doc(
Computes the induced subgraph given by the specified vertices in the given graph.

This function represents subgraphs as a packed adjacency list, that is, an adjacency
list with the list of neighbours for each vertex concatenated. As this representation
uses canonical vertex identities, the output neighbours of this function will
be relabelled.

Parameters
----------
vertex: The vertices spanning the subgraph. Must be in sorted order.
neighbours: The packed adjacency list neighbours.
lengths: The packed adjacency list lengths.
offsets: The packed adjacency list offsets.

Returns
-------
subgraph_neighbours: The packed adjacency list representing the induced subgraph.
subgraph_lengths: The lengths of the subarrays.
subgraph_offsets: The offsets of the subarray.
)doc");

REGISTER_KERNEL_BUILDER(Name("GetInducedSubgraph").Device("CPU"), GetInducedSubgraphOp);