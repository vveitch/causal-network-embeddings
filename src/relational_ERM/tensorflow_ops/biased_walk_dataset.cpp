#include <algorithm>
#include <numeric>
#include <stack>
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/dataset.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/util/guarded_philox_random.h>
#include <tensorflow/core/lib/random/random_distributions.h>
#include <tensorflow/core/util/work_sharder.h>

using namespace tensorflow;

template<typename IndexType = int32, typename FloatType = float>
struct AliasDrawData {
    std::unique_ptr<IndexType[]> draw_lengths_;
    std::unique_ptr<IndexType[]> draw_offsets_;
    std::unique_ptr<FloatType[]> draw_probabilities_;
    std::unique_ptr<IndexType[]> draw_outcomes_;
    IndexType draw_total_size_;
};

template<typename IndexType, typename FloatType>
void setup_alias_sample_impl(IndexType num_outcomes, FloatType* probabilities, IndexType* outcomes) {
    std::stack<IndexType> smaller_outcomes;
    std::stack<IndexType> larger_outcomes;

    for(IndexType outcome = 0; outcome < num_outcomes; ++outcome) {
        probabilities[outcome] *= num_outcomes;
        if(probabilities[outcome] < 1) {
            smaller_outcomes.push(outcome);
        }
        else {
            larger_outcomes.push(outcome);
        }
    }

    while((!smaller_outcomes.empty()) && (!larger_outcomes.empty())) {
        auto small = smaller_outcomes.top();
        auto large = larger_outcomes.top();

        smaller_outcomes.pop();
        larger_outcomes.pop();

        outcomes[small] = large;
        probabilities[large] -= 1 - probabilities[small];

        if(probabilities[large] < 1) {
            smaller_outcomes.push(large);
        }
        else {
            larger_outcomes.push(large);
        }
    }
}

template<typename IndexType, typename FloatType, typename Rng>
IndexType alias_draw(Rng& rng, IndexType num_outcomes, const FloatType* probabilities, const IndexType* outcomes) {
    random::UniformDistribution<Rng, IndexType> mixture_outcome(0, num_outcomes);
    random::UniformDistribution<Rng, FloatType> uniform;

    IndexType kk = mixture_outcome(&rng)[0];

    if(uniform(&rng)[0] < probabilities[kk]) {
        return kk;
    }
    else {
        return outcomes[kk];
    }
}


template<typename IndexType, typename I1, typename I2, typename O1, typename FloatType>
void compute_transition_probabilities(
    IndexType previous_node,
    FloatType p, FloatType q,
    I1 current_neighbours_begin,
    I1 current_neighbours_end,
    I2 previous_neighbours_begin,
    I2 previous_neighbours_end,
    O1 output_begin) {
    
    auto output_current = output_begin;

    // first compute unormalized probabilities
    for(; current_neighbours_begin != current_neighbours_end; ++current_neighbours_begin, ++output_current) {

        // first case, the target vertex is the same as the previous node.
        if(*current_neighbours_begin == previous_node) {
            *output_current = 1;
            continue;
        }

        // second case, the target vertex is a neighbour of the previous node.
        previous_neighbours_begin = std::lower_bound(previous_neighbours_begin, previous_neighbours_end, *current_neighbours_begin);

        if (previous_neighbours_begin != previous_neighbours_end &&
            *previous_neighbours_begin == *current_neighbours_begin) {
            *output_current = 1 / p;
            continue;
        }

        // last case, the target vertex is exactly distance 2 away from previous node.
        *output_current = 1 / q;
    }

    // next pass over again to compute normalized probabilities.
    auto normalization = std::accumulate(output_begin, output_current, 0);
    std::transform(output_begin, output_current, output_begin, [=](typename std::iterator_traits<O1>::value_type x) {
        return x / normalization;
    });
}


template<typename GraphIndexType, typename IndexType, typename FloatType>
void precompute_biased_walk(
    const Tensor& neighbours_tensor, const Tensor& lengths_tensor, const Tensor& offsets_tensor,
    FloatType p, FloatType q, OpKernelContext* ctx, AliasDrawData<IndexType, FloatType>& output) {

    // Pre-compute the required lengths, offsets and sizes.
    auto num_neighbours = neighbours_tensor.NumElements();
    auto num_vertex = lengths_tensor.NumElements();
    int64 total_size = 0;

    output.draw_lengths_.reset(new IndexType[num_neighbours]);
    output.draw_offsets_.reset(new IndexType[num_neighbours]);

    auto const& neighbours = neighbours_tensor.flat<GraphIndexType>();
    auto const& lengths = lengths_tensor.flat<GraphIndexType>();
    auto const& offsets = offsets_tensor.flat<GraphIndexType>();

    for(int64 i = 0; i < num_neighbours; ++i) {
        auto current_length = lengths(neighbours(i));

        OP_REQUIRES(ctx, current_length > 0,
                    errors::InvalidArgument("Element of length was negative."));

        output.draw_lengths_[i] = current_length;
        output.draw_offsets_[i] = total_size;
        total_size += current_length;
    }

    output.draw_total_size_ = static_cast<IndexType>(total_size);
    output.draw_outcomes_.reset(new IndexType[total_size]);
    output.draw_probabilities_.reset(new FloatType[total_size]);

    auto work = [&](int64 previous_vertex_start, int64 previous_vertex_end) {
        for(auto previous_vertex = previous_vertex_start; previous_vertex < previous_vertex_end; ++previous_vertex) {
            auto num_previous_neighbours = lengths(previous_vertex);
            auto previous_vertex_offset = offsets(previous_vertex);

            for(IndexType current_vertex_offset = 0; current_vertex_offset < num_previous_neighbours; ++current_vertex_offset) {
                auto edge_index = previous_vertex_offset + current_vertex_offset;
                auto current_vertex = neighbours(edge_index);

                compute_transition_probabilities(
                    previous_vertex, p, q,
                    neighbours.data() + offsets(current_vertex),
                    neighbours.data() + offsets(current_vertex) + lengths(current_vertex),
                    neighbours.data() + offsets(previous_vertex),
                    neighbours.data() + offsets(previous_vertex) + lengths(previous_vertex),
                    output.draw_probabilities_.get() + output.draw_offsets_[edge_index]);
                
                setup_alias_sample_impl(
                    output.draw_lengths_[edge_index],
                    output.draw_probabilities_.get() + output.draw_offsets_[edge_index],
                    output.draw_outcomes_.get() + output.draw_offsets_[edge_index]);
            }
        }
    };

    const double averageNumberNeighbours = static_cast<double>(num_neighbours) / static_cast<double>(num_vertex);
    const auto transitionProbabilityCost = 5 * averageNumberNeighbours * log(averageNumberNeighbours);
    const auto setupAliasSampleCost = 10 * averageNumberNeighbours;
    const auto costPerUnitEdge = transitionProbabilityCost + setupAliasSampleCost;
    const auto costPerUnit = static_cast<int64>(averageNumberNeighbours * costPerUnitEdge);

    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();

    Shard(worker_threads->num_threads, worker_threads->workers, num_vertex, costPerUnit, work);
}


template<typename IndexType, typename DrawIndexType, typename FloatType>
void biased_walk_sample(GuardedPhiloxRandom& guarded_generator, IndexType num_vertex, IndexType walk_length,
                        const IndexType* graph_neighbours, const IndexType* graph_lengths, const IndexType* graph_offsets,
                        const DrawIndexType* draw_lengths, const DrawIndexType* draw_offsets,
                        const FloatType* draw_probabilities, const DrawIndexType* draw_outcomes,
                        IndexType* output_walk) {
    if(walk_length == 0) {
        return;
    }

    auto generator = guarded_generator.ReserveRandomOutputs(walk_length * 2 + 1, 256);
    random::UniformDistribution<random::PhiloxRandom, IndexType> random_node_distribution(0, num_vertex);

    // bootstrap the walk.
    IndexType current_length = 1;
    IndexType current_vertex = random_node_distribution(&generator)[0];
    output_walk[0] = current_vertex;

    if(graph_lengths[current_vertex] == 0) {
        // we are at a singleton vertex.
        std::fill(output_walk + 1, output_walk + walk_length, -1);
        return;
    }

    if(walk_length == 1) {
        // we're done here.
        return;
    }

    // sample second step uniformly
    auto previous_vertex = current_vertex;
    random::UniformDistribution<random::PhiloxRandom, IndexType> random_neighbour(0, graph_lengths[current_vertex]);
    IndexType current_offset = random_neighbour(&generator)[0];
    current_vertex = graph_neighbours[graph_offsets[previous_vertex] + current_offset];
    output_walk[1] = current_vertex;
    current_length += 1;

    // main loop to do the 2-markovian sample
    while(current_length < walk_length) {
        // At a given step, previous_vertex refers to the vertex where we are coming from
        // and current_vertex the vertex we are currently at. This can be represented
        // as an edge, whose index is computed below.

        // Note that current_offset is the offset of the current vertex in the neighbours
        // of the previous vertex.
        auto current_edge = graph_offsets[previous_vertex] + current_offset;

        auto next_offset = alias_draw(
            generator,
            draw_lengths[current_edge],
            draw_probabilities + draw_offsets[current_edge],
            draw_outcomes + draw_offsets[current_edge]);
        
        auto next_vertex = graph_neighbours[graph_offsets[current_vertex] + next_offset];
        output_walk[current_length] = next_vertex;

        // update state to continue iteration.
        current_length += 1;
        previous_vertex = current_vertex;
        current_vertex = next_vertex;
        current_offset = next_offset;
    }
}


class BiasedWalkDatasetOp : public DatasetOpKernel {
public:
    explicit BiasedWalkDatasetOp(OpKernelConstruction* ctx): DatasetOpKernel(ctx) {
    }

    void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
        const Tensor* neighbours;
        const Tensor* lengths;
        const Tensor* offsets;

        int32 walk_length;
        float p;
        float q;

        int64 seed;
        int64 seed2;

        OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed", &seed));
        OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed2", &seed2));

        OP_REQUIRES_OK(ctx, ctx->input("neighbours", &neighbours));
        OP_REQUIRES_OK(ctx, ctx->input("lengths", &lengths));
        OP_REQUIRES_OK(ctx, ctx->input("offsets", &offsets));

        OP_REQUIRES_OK(ctx, ParseScalarArgument<int32>(ctx, "walk_length", &walk_length));
        OP_REQUIRES_OK(ctx, ParseScalarArgument<float>(ctx, "p", &p));
        OP_REQUIRES_OK(ctx, ParseScalarArgument<float>(ctx, "q", &q));

        AliasDrawData<int64, float> draw_data;
        precompute_biased_walk<int32>(*neighbours, *lengths, *offsets, p, q, ctx, draw_data);

        *output = new Dataset(ctx, seed, seed2, walk_length, p, q, *neighbours, *lengths, *offsets, std::move(draw_data));
    }

private:
    TF_DISALLOW_COPY_AND_ASSIGN(BiasedWalkDatasetOp);

    class Dataset: public DatasetBase {
    public:
        explicit Dataset(OpKernelContext* ctx, int64 seed, int64 seed2,
                         int32 walk_length, float p, float q,
                         Tensor neighbours, Tensor lengths, Tensor offsets,
                         AliasDrawData<int64, float> draw_data):
            DatasetBase(DatasetContext(ctx)), neighbours_(neighbours), lengths_(lengths), offsets_(offsets),
            walk_length_(walk_length), seed_(seed), seed2_(seed2), draw_data_(std::move(draw_data))
        {
        }

        std::unique_ptr<IteratorBase> MakeIteratorInternal(const string& prefix) const override {
            return std::unique_ptr<IteratorBase>(
                new Iterator({this, strings::StrCat(prefix, "::BiasedWalkSample")})
            );
        }

        const DataTypeVector& output_dtypes() const override {
            static DataTypeVector* dtypes = new DataTypeVector({DT_INT32});
            return *dtypes;
        }

        const std::vector<PartialTensorShape>& output_shapes() const override {
            static std::vector<PartialTensorShape>* shapes = new std::vector<PartialTensorShape>({{-1}});
            return *shapes;
        }

        string DebugString() const override {
            return "BiasedWalkDatasetOp::Dataset";
        }
    
    protected:
        Status AsGraphDefInternal(SerializationContext* ctx, DatasetGraphDefBuilder* b, Node** output) const override {
            Node* neighbours = nullptr;
            TF_RETURN_IF_ERROR(b->AddTensor(neighbours_, &neighbours));

            Node* lengths = nullptr;
            TF_RETURN_IF_ERROR(b->AddTensor(lengths_, &lengths));

            Node* offsets = nullptr;
            TF_RETURN_IF_ERROR(b->AddTensor(offsets_, &offsets));

            Node* walk_length = nullptr;
            TF_RETURN_IF_ERROR(b->AddScalar(walk_length_, &walk_length));

            Node* seed = nullptr;
            TF_RETURN_IF_ERROR(b->AddScalar(seed_, &seed));

            Node* seed2 = nullptr;
            TF_RETURN_IF_ERROR(b->AddScalar(seed2_, &seed2));

            return Status::OK();
        }
    private:
        const Tensor neighbours_;
        const Tensor lengths_;
        const Tensor offsets_;
        const AliasDrawData<int64, float> draw_data_;
        const int32 walk_length_;
        const int64 seed_;
        const int64 seed2_;

        class Iterator: public DatasetIterator<Dataset> {
        public:
            explicit Iterator(const Params& params):
                DatasetIterator<Dataset>(params) {
                    generator_.Init(dataset()->seed_, dataset()->seed2_);
                }

            Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors, bool* end_of_sequence) override {
                Tensor output_tensor(ctx->allocator({}), DT_INT32, {dataset()->walk_length_});

                biased_walk_sample(
                    generator_, static_cast<int32>(dataset()->offsets_.NumElements()), dataset()->walk_length_,
                    dataset()->neighbours_.flat<int32>().data(),
                    dataset()->lengths_.flat<int32>().data(),
                    dataset()->offsets_.flat<int32>().data(),
                    dataset()->draw_data_.draw_lengths_.get(), dataset()->draw_data_.draw_offsets_.get(),
                    dataset()->draw_data_.draw_probabilities_.get(), dataset()->draw_data_.draw_outcomes_.get(),
                    output_tensor.flat<int32>().data());
                
                out_tensors->emplace_back(std::move(output_tensor));

                return Status::OK();
            }
        private:
            GuardedPhiloxRandom generator_;
        };
    };
};

REGISTER_OP("BiasedWalkDataset")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Input("neighbours: int32")
    .Input("offsets: int32")
    .Input("lengths: int32")
    .Input("walk_length: int32")
    .Input("p: float32")
    .Input("q: float32")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &input_shape));

        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &input_shape));

        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &input_shape));

        return shape_inference::ScalarShape(c);
    });

REGISTER_KERNEL_BUILDER(Name("BiasedWalkDataset").Device("CPU"), BiasedWalkDatasetOp);