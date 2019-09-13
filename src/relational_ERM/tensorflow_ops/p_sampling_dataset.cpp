#include <algorithm>
#include <numeric>
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/dataset.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/util/guarded_philox_random.h>
#include <tensorflow/core/lib/random/random_distributions.h>

using namespace tensorflow;

template<typename InputArray1, typename InputArray2, typename OutputArray, typename IndexType>
IndexType set_intersection_index(InputArray1 const& input1, IndexType length1,
                                 InputArray2 const& input2, IndexType length2,
                                 OutputArray& output) {
    IndexType i = 0, j = 0, k = 0;

    while(i < length1 && j < length2) {
        if(input1[i] == input2[j]) {
            output[k] = i;
            ++i;
            ++j;
            ++k;
        }
        else if(input1[i] < input2[j]) {
            ++i;
        }
        else {
            ++j;
        }
    }

    return k;
}


template<typename Distribution, typename It>
void fill_random(random::PhiloxRandom& gen, Distribution& dist, It first, It last) {
    const int samples_per_call = Distribution::kResultElementCount;

    while(std::distance(first, last) >= samples_per_call) {
        const auto samples = dist(&gen);
        std::copy(&samples[0], &samples[0] + samples_per_call, first);
        std::advance(first, samples_per_call);
    }

    const auto samples_last = dist(&gen);
    std::copy(&samples_last[0], &samples_last[0] + std::distance(first, last), first);
}

template<typename IndexType>
void p_sample(IndexType num_vertex, const IndexType* neighbours, const IndexType* lengths, const IndexType* offsets,
              float p, IteratorContext* ctx, GuardedPhiloxRandom& guarded_generator,
              Tensor* out_neighbours, Tensor* out_lengths, Tensor* out_offsets, Tensor* out_vertex_index) {

    auto generator = guarded_generator.ReserveRandomOutputs(num_vertex, 64);
    random::UniformDistribution<random::PhiloxRandom, float> uniform_float;

    std::vector<float> vertex_uniform_sample(num_vertex);
    fill_random(generator, uniform_float, vertex_uniform_sample.begin(), vertex_uniform_sample.end());

    std::vector<IndexType> candidate_vertices;

    for(IndexType i = 0; i < num_vertex; ++i) {
        if(vertex_uniform_sample[i] < p) {
            candidate_vertices.push_back(i);
        }
    }

    auto num_candidate = static_cast<IndexType>(candidate_vertices.size());

    std::vector<IndexType> vertex_index;
    std::vector<IndexType> sampled_lengths;
    std::vector<IndexType> sampled_neighbours;

    std::vector<IndexType> vertex_index_inverse(num_vertex);
    std::vector<IndexType> temp_intersection(num_candidate);

    IndexType num_vertex_sampled = 0;
    
    for(IndexType i = 0; i < num_candidate; ++i) {
        auto vertex = candidate_vertices[i];
        auto current_neighbours = neighbours + offsets[vertex];

        // Detect if vertex is a singleton
        auto num_remaining = set_intersection_index(
            current_neighbours, lengths[vertex],
            candidate_vertices.begin(), num_candidate, temp_intersection);
        
        // temp_intersection now indexes into the offset neighbour array
        // indicating which have been selected. We merge this information
        // if more than one has been selected.
        if (num_remaining == 0) {
            // no indices in common
            continue;
        }

        // Maintain the current mapping of vertex indices.
        sampled_lengths.push_back(num_remaining);
        vertex_index_inverse[vertex] = num_vertex_sampled;
        vertex_index.push_back(vertex);
        num_vertex_sampled += 1;

        for(IndexType j = 0; j < num_remaining; ++j) {
            sampled_neighbours.push_back(current_neighbours[temp_intersection[j]]);
        }
    }

    // Re-label vertices in subgraph.
    std::transform(
        sampled_neighbours.begin(), sampled_neighbours.end(), sampled_neighbours.begin(),
        [&](IndexType v) {
            return vertex_index_inverse[v];
        });
    
    auto num_sampled_neighbours = static_cast<IndexType>(sampled_neighbours.size());
    
    *out_neighbours = Tensor(ctx->allocator({}), DT_INT32, {num_sampled_neighbours});
    *out_lengths = Tensor(ctx->allocator({}), DT_INT32, {num_vertex_sampled});
    *out_offsets = Tensor(ctx->allocator({}), DT_INT32, {num_vertex_sampled});
    *out_vertex_index = Tensor(ctx->allocator({}), DT_INT32, {num_vertex_sampled});

    out_offsets->flat<IndexType>().data()[0] = 0;
    std::partial_sum(sampled_lengths.begin(), sampled_lengths.end() - 1, out_offsets->flat<IndexType>().data() + 1);

    std::copy(sampled_neighbours.begin(), sampled_neighbours.end(), out_neighbours->flat<IndexType>().data());
    std::copy(sampled_lengths.begin(), sampled_lengths.end(), out_lengths->flat<IndexType>().data());
    std::copy(vertex_index.begin(), vertex_index.end(), out_vertex_index->flat<IndexType>().data());
}

class PSamplingDatasetOp : public DatasetOpKernel {
public:
    explicit PSamplingDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    }


    void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
        const Tensor* neighbours;
        const Tensor* lengths;
        const Tensor* offsets;
        float p;

        int64 seed;
        int64 seed2;

        OP_REQUIRES_OK(ctx, ctx->input("neighbours", &neighbours));
        OP_REQUIRES_OK(ctx, ctx->input("lengths", &lengths));
        OP_REQUIRES_OK(ctx, ctx->input("offsets", &offsets));
        OP_REQUIRES_OK(ctx, ParseScalarArgument<float>(ctx, "p", &p));

        OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed", &seed));
        OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed2", &seed2));

        *output = new Dataset(ctx, seed, seed2, p, *neighbours, *lengths, *offsets);
    }
private:
    TF_DISALLOW_COPY_AND_ASSIGN(PSamplingDatasetOp);


    class Dataset: public DatasetBase {
    public:
        explicit Dataset(OpKernelContext* ctx, int64 seed, int64 seed2, float p,
                         Tensor neighbours, Tensor lengths, Tensor offsets):
            DatasetBase(DatasetContext(ctx)), neighbours_(std::move(neighbours)),
            lengths_(std::move(lengths)), offsets_(std::move(offsets)),
            p_(p), seed_(seed), seed2_(seed2)
        {
        }

        std::unique_ptr<IteratorBase> MakeIteratorInternal(const string& prefix) const override {
            return std::unique_ptr<IteratorBase>(
                new Iterator({this, strings::StrCat(prefix, "::PSample")})
            );
        }

        const DataTypeVector& output_dtypes() const override {
            static DataTypeVector* dtypes = new DataTypeVector({DT_INT32, DT_INT32, DT_INT32, DT_INT32});
            return *dtypes;
        }

        const std::vector<PartialTensorShape>& output_shapes() const override {
            static std::vector<PartialTensorShape>* shapes = new std::vector<PartialTensorShape>({{-1}, {-1}, {-1}, {-1}});
            return *shapes;
        }

        string DebugString() const override {
            return "PSamplingDatasetOp::Dataset";
        }
    
    protected:
        Status AsGraphDefInternal(SerializationContext* ctx, DatasetGraphDefBuilder* b, Node** output) const override {
            Node* neighbours = nullptr;
            TF_RETURN_IF_ERROR(b->AddTensor(neighbours_, &neighbours));
            Node* lengths = nullptr;
            TF_RETURN_IF_ERROR(b->AddTensor(lengths_, &lengths));
            Node* offsets = nullptr;
            TF_RETURN_IF_ERROR(b->AddTensor(offsets_, &offsets));

            Node* p = nullptr;
            TF_RETURN_IF_ERROR(b->AddScalar(p_, &p));
            Node* seed = nullptr;
            TF_RETURN_IF_ERROR(b->AddScalar(seed_, &seed));
            Node* seed2 = nullptr;
            TF_RETURN_IF_ERROR(b->AddScalar(seed2_, &seed2));

            return Status::OK();
        }
    private:
        Tensor neighbours_;
        Tensor lengths_;
        Tensor offsets_;
        float p_;
        int64 seed_;
        int64 seed2_;

        class Iterator: public DatasetIterator<Dataset> {
        public:
            explicit Iterator(const Params& params):
                DatasetIterator<Dataset>(params) {
                    generator_.Init(dataset()->seed_, dataset()->seed2_);
                }

            Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors, bool* end_of_sequence) override {
                Tensor out_neighbours;
                Tensor out_lengths;
                Tensor out_offsets;
                Tensor out_vertex_index;

                p_sample(static_cast<int32>(dataset()->lengths_.NumElements()),
                         dataset()->neighbours_.flat<int32>().data(),
                         dataset()->lengths_.flat<int32>().data(),
                         dataset()->offsets_.flat<int32>().data(),
                         dataset()->p_, ctx,
                         generator_,
                         &out_neighbours,
                         &out_lengths,
                         &out_offsets,
                         &out_vertex_index);
                
                // place outputs alphabetically to ensure correct
                // unpacking on the python side.
                out_tensors->reserve(4);
                out_tensors->emplace_back(std::move(out_lengths));
                out_tensors->emplace_back(std::move(out_neighbours));
                out_tensors->emplace_back(std::move(out_offsets));
                out_tensors->emplace_back(std::move(out_vertex_index));

                return Status::OK();
            }
        private:
            GuardedPhiloxRandom generator_;
        };
    };
};


REGISTER_OP("PSamplingDataset")
    .Input("neighbours: int32")
    .Input("offsets: int32")
    .Input("lengths: int32")
    .Input("p: float32")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input_shape));

        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &input_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &input_shape));

        return shape_inference::ScalarShape(c);
    });

REGISTER_KERNEL_BUILDER(Name("PSamplingDataset").Device("CPU"), PSamplingDatasetOp);