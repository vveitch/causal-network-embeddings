#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/lib/core/errors.h>


using namespace tensorflow;


/*! Functor which implements the conversion from a packed adjacency list
 *  to a redundant edge list.
 */
template<typename IndexType>
struct PositiveRedundantEdgeListFunctor {
    void operator()(typename TTypes<const IndexType, 1>::Vec const& neighbours,
                    typename TTypes<const IndexType, 1>::Vec const& lengths,
                    typename TTypes<IndexType, 2>::Matrix& out_edges) const {

        auto num_vertex = lengths.dimension(0);
        int edge_index = 0;

        for(int source_vertex = 0; source_vertex < num_vertex; ++source_vertex) {
            auto num_neighbours = lengths(source_vertex);

            for (int vertex_offset = 0; vertex_offset < num_neighbours; ++vertex_offset, ++edge_index) {
                out_edges(edge_index, 0) = source_vertex;
                out_edges(edge_index, 1) = neighbours(edge_index);
            }
        }
    }
};


/*! Functor which implements the conversion from a symmetric packed adjacency
 *  list to a canonical edge list.
 */
template<typename IndexType>
struct PositiveCanonicalEdgeListFunctor {
    void operator()(typename TTypes<const IndexType, 1>::Vec const& neighbours,
                    typename TTypes<const IndexType, 1>::Vec const& lengths,
                    typename TTypes<IndexType, 2>::Matrix& out_edges) const {
        
        auto num_vertex = lengths.dimension(0);
        int edge_index_in = 0;
        int edge_index_out = 0;

        for(int source_vertex = 0; source_vertex < num_vertex; ++source_vertex) {
            auto num_neighbours = lengths(source_vertex);

            for (int vertex_offset = 0; vertex_offset < num_neighbours; ++vertex_offset, ++edge_index_in) {
                auto target_vertex = neighbours(edge_index_in);

                if (source_vertex > target_vertex) {
                    continue;
                }

                out_edges(edge_index_out, 0) = source_vertex;
                out_edges(edge_index_out, 1) = target_vertex;
                ++edge_index_out;
            }
        }
    }
};

/*! This operation implements a method to produce an edge list from a packed
 *  adjacency list, with all positive edges preserved and all negative edges
 *  removed.
 * 
 *  The type of edge list produced may be controlled by the `redundant` attribute.
 *  If true, no assumption is made on the packed adjacency list and the output
 *  edge list is redundant (i.e. it contains both edges (a, b) and (b, a)).
 *  If false, the adjacency list is assumed to be symmetric (i.e. a being a neighbour of b
 *  must imply b a neighbour of a), and the computed edge list is canonical, i.e. only
 *  edges (a, b), with a < b appear in the edge list.
 */
class AdjacencyToEdgeListOp : public OpKernel {
public:
    explicit AdjacencyToEdgeListOp(OpKernelConstruction* ctx): OpKernel(ctx), redundant_(false) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("redundant", &this->redundant_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* neighbours;
        const Tensor* lengths;

        OP_REQUIRES_OK(ctx, ctx->input("neighbours", &neighbours));
        OP_REQUIRES_OK(ctx, ctx->input("lengths", &lengths));

        OP_REQUIRES(ctx, TensorShapeUtils::IsVector(neighbours->shape()),
                    errors::InvalidArgument("neighbours must be a vector."));
        OP_REQUIRES(ctx, TensorShapeUtils::IsVector(lengths->shape()),
                    errors::InvalidArgument("lengths must be a vector."));

        auto num_edges_in = neighbours->shape().dim_size(0);
        auto num_vertex = lengths->shape().dim_size(0);

        auto num_edges_out = num_edges_in;

        if (!redundant_) {
            num_edges_out /= 2;
        }

        Tensor* output_edges;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {num_edges_out, 2}, &output_edges));

        auto const& neighbours_flat = neighbours->vec<int32>();
        auto const& lengths_flat = lengths->vec<int32>();
        auto output_edges_mat = output_edges->matrix<int32>();

        if (redundant_) {
            PositiveRedundantEdgeListFunctor<int32> functor;
            functor(neighbours_flat, lengths_flat, output_edges_mat);
        } else {
            PositiveCanonicalEdgeListFunctor<int32> functor;
            functor(neighbours_flat, lengths_flat, output_edges_mat);
        }
    }
private:
    bool redundant_;
};

REGISTER_OP("AdjacencyToEdgeList")
    .Input("neighbours: int32")
    .Input("lengths: int32")
    .Attr("redundant: bool")
    .Output("edge_list: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape;

        bool redundant;
        TF_RETURN_IF_ERROR(c->GetAttr("redundant", &redundant));

        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));

        auto num_edges = c->Dim(c->input(0), 0);

        if (redundant) {
            c->Divide(num_edges, 2, true, &num_edges);
        }

        c->set_output(0, c->Matrix(num_edges, 2));

        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("AdjacencyToEdgeList").Device("CPU"), AdjacencyToEdgeListOp);

template<typename IndexType>
struct PosNegRedundantEdgeListFunctor {
    void operator()(typename TTypes<const IndexType, 1>::Vec const& neighbours,
                    typename TTypes<const IndexType, 1>::Vec const& lengths,
                    typename TTypes<IndexType, 2>::Matrix& out_edges_pos,
                    typename TTypes<IndexType, 2>::Matrix& out_edges_neg,
                    OpKernelContext* context) const {

        auto num_vertex = lengths.dimension(0);
        int edge_index_pos = 0;
        int edge_index_neg = 0;
        int edge_index_source = 0;

        for(IndexType source_vertex = 0; source_vertex < num_vertex; ++source_vertex) {
            auto num_neighbours = lengths(source_vertex);

            OP_REQUIRES(context, num_neighbours <= num_vertex,
                        errors::InvalidArgument("Lengths of subarray larger than number of vertices."));

            IndexType current_offset = 0;

            for (IndexType target_vertex = 0; target_vertex < num_vertex; ++target_vertex) {
                // for each potential vertex, decide whether it is an edge or a non-edge.
                // our current target vertex is always less or equal than the next neighbour,
                // except in the case where we have exhausted all existing neighbours.

                if (target_vertex == source_vertex) {
                    // no self-edges, so skip this case.
                    OP_REQUIRES(context, current_offset == num_neighbours || target_vertex != neighbours(edge_index_source),
                                errors::InvalidArgument("Self edge in graph neighbours."));

                    continue;
                }

                if (current_offset < num_neighbours && target_vertex == neighbours(edge_index_source)) {
                    // matching case, add positive edge
                    out_edges_pos(edge_index_pos, 0) = source_vertex;
                    out_edges_pos(edge_index_pos, 1) = target_vertex;
                    ++edge_index_pos;
                    ++edge_index_source;
                    ++current_offset;
                } else {
                    // non-matching case, add negative edge.
                    out_edges_neg(edge_index_neg, 0) = source_vertex;
                    out_edges_neg(edge_index_neg, 1) = target_vertex;
                    ++edge_index_neg;
                }
            }

            OP_REQUIRES(context, current_offset == num_neighbours,
                        errors::Internal("Mismatch in number of neighbours."));
        }

        OP_REQUIRES(context, edge_index_pos == out_edges_pos.dimension(0),
                    errors::Internal("Mismatch in expected positive edge count."));
        
        OP_REQUIRES(context, edge_index_neg == out_edges_neg.dimension(0),
                    errors::Internal("Mismatch in expected negative edge count."));
    }
};


template<typename IndexType>
struct PosNegCanonicalEdgeListFunctor {
    void operator()(typename TTypes<const IndexType, 1>::Vec const& neighbours,
                    typename TTypes<const IndexType, 1>::Vec const& lengths,
                    typename TTypes<IndexType, 2>::Matrix& out_edges_pos,
                    typename TTypes<IndexType, 2>::Matrix& out_edges_neg,
                    OpKernelContext* context) const {

        auto num_vertex = lengths.dimension(0);
        int edge_index_pos = 0;
        int edge_index_neg = 0;
        int edge_index_source = 0;

        for(IndexType source_vertex = 0; source_vertex < num_vertex; ++source_vertex) {
            auto num_neighbours = lengths(source_vertex);

            IndexType current_offset = 0;

            // advance past all neighbours that are smaller than the current source.
            while(current_offset < num_neighbours && neighbours(edge_index_source) < source_vertex + 1) {
                ++edge_index_source;
                ++current_offset;
            }

            for (IndexType target_vertex = source_vertex + 1; target_vertex < num_vertex; ++target_vertex) {
                // for each potential vertex, decide whether it is an edge or a non-edge.
                // our current target vertex is always less or equal than the next neighbour,
                // except in the case where we have exhausted all existing neighbours.

                if (current_offset < num_neighbours && target_vertex == neighbours(edge_index_source)) {
                    // matching case, add positive edge
                    out_edges_pos(edge_index_pos, 0) = source_vertex;
                    out_edges_pos(edge_index_pos, 1) = target_vertex;
                    ++edge_index_pos;
                    ++edge_index_source;
                    ++current_offset;

                } else {
                    // non-matching case, add negative edge.
                    out_edges_neg(edge_index_neg, 0) = source_vertex;
                    out_edges_neg(edge_index_neg, 1) = target_vertex;
                    ++edge_index_neg;
                }
            }

            OP_REQUIRES(context, current_offset == num_neighbours,
                        errors::Internal("Mismatch in number of neighbours."));
        }

        OP_REQUIRES(context, edge_index_pos == out_edges_pos.dimension(0),
                    errors::Internal("Mismatch in expected positive edge count."));
        
        OP_REQUIRES(context, edge_index_neg == out_edges_neg.dimension(0),
                    errors::Internal("Mismatch in expected negative edge count."));
    }
};

class AdjacencyToPosNegEdgeListOp : public OpKernel {
public:
    explicit AdjacencyToPosNegEdgeListOp(OpKernelConstruction* ctx): OpKernel(ctx), redundant_(false) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("redundant", &this->redundant_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* neighbours;
        const Tensor* lengths;

        OP_REQUIRES_OK(ctx, ctx->input("neighbours", &neighbours));
        OP_REQUIRES_OK(ctx, ctx->input("lengths", &lengths));

        auto num_edges_in = neighbours->shape().dim_size(0);
        auto num_vertex = lengths->shape().dim_size(0);

        auto num_edges_out_pos = num_edges_in;
        auto num_edges_out_neg = num_vertex * (num_vertex - 1) - num_edges_out_pos;

        if (!redundant_) {
            num_edges_out_pos /= 2;
            num_edges_out_neg /= 2;
        }

        Tensor* output_edges_pos;
        Tensor* output_edges_neg;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {num_edges_out_pos, 2}, &output_edges_pos));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {num_edges_out_neg, 2}, &output_edges_neg));

        auto const& neighbours_flat = neighbours->vec<int32>();
        auto const& lengths_flat = lengths->vec<int32>();
        auto output_edges_pos_mat = output_edges_pos->matrix<int32>();
        auto output_edges_neg_mat = output_edges_neg->matrix<int32>();

        if (redundant_) {
            PosNegRedundantEdgeListFunctor<int32> functor;
            functor(neighbours_flat, lengths_flat, output_edges_pos_mat, output_edges_neg_mat, ctx);
        } else  {
            PosNegCanonicalEdgeListFunctor<int32> functor;
            functor(neighbours_flat, lengths_flat, output_edges_pos_mat, output_edges_neg_mat, ctx);
        }
    }
private:
    bool redundant_;
};


REGISTER_OP("AdjacencyToPosNegEdgeList")
    .Input("neighbours: int32")
    .Input("lengths: int32")
    .Attr("redundant: bool")
    .Output("edge_list_pos: int32")
    .Output("edge_list_neg: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape;

        bool redundant;
        TF_RETURN_IF_ERROR(c->GetAttr("redundant", &redundant));

        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));

        auto num_edges = c->Dim(c->input(0), 0);
        auto num_vertex = c->Dim(c->input(1), 0);
        shape_inference::DimensionHandle total_edges;
        shape_inference::DimensionHandle num_edges_neg;

        c->Multiply(num_vertex, num_vertex, &total_edges);
        c->Subtract(total_edges, num_vertex, &total_edges);
        c->Subtract(total_edges, num_edges, &num_edges_neg);

        if (redundant) {
            c->Divide(num_edges, 2, true, &num_edges);
            c->Divide(num_edges_neg, 2, true, &num_edges_neg);
        }

        c->set_output(0, c->Matrix(num_edges, 2));
        c->set_output(1, c->Matrix(num_edges_neg, 2));

        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("AdjacencyToPosNegEdgeList").Device("CPU"), AdjacencyToPosNegEdgeListOp);