#include <algorithm>
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/dataset.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/util/guarded_philox_random.h>
#include <tensorflow/core/lib/random/random_distributions.h>

using namespace tensorflow;


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
void uniform_edge_sample(IndexType sample_size,
                         GuardedPhiloxRandom& guarded_generator,
                         typename TTypes<IndexType, 1>::ConstVec const& neighbours,
                         typename TTypes<IndexType, 1>::ConstVec const& offsets,
                         typename TTypes<IndexType, 2>::Matrix out_sample) {
    auto n_edges = static_cast<IndexType>(neighbours.dimension(0));
    auto num_vertex = static_cast<IndexType>(offsets.dimension(0));
    auto generator = guarded_generator.ReserveRandomOutputs(sample_size, 128);

    random::UniformDistribution<random::PhiloxRandom, IndexType> uniform(0, n_edges);

    // we draw a bunch of uniform indices, which represent each edge
    // as a location in the neighbours array.
    std::vector<IndexType> edge_index(sample_size);
    fill_random(generator, uniform, edge_index.begin(), edge_index.end());
    std::sort(edge_index.begin(), edge_index.end());


    // the neighbours array contains the target of each edge at a given location
    // the source is given implicitly through the offsets tensor.
    
    auto current_source_index = 1;

    for(IndexType i = 0; i < sample_size; ++i) {
        auto current_edge = edge_index[i];

        while (current_source_index < num_vertex && current_edge >= offsets(current_source_index)) {
            ++current_source_index;
        }

        out_sample(i, 0) = current_source_index - 1;
        out_sample(i, 1) = neighbours(current_edge);
    }
}


class UniformEdgeDatasetOp : public DatasetOpKernel {
public:
    explicit UniformEdgeDatasetOp(OpKernelConstruction* ctx): DatasetOpKernel(ctx) {
    }


    void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
        const Tensor* neighbours;
        const Tensor* lengths;
        const Tensor* offsets;
        int32 n;
        int64 seed, seed2;

        OP_REQUIRES_OK(ctx, ctx->input("neighbours", &neighbours));
        OP_REQUIRES_OK(ctx, ctx->input("lengths", &lengths));
        OP_REQUIRES_OK(ctx, ctx->input("offsets", &offsets));

        OP_REQUIRES_OK(ctx, ParseScalarArgument<int32>(ctx, "n", &n));
        OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "n", &seed));
        OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "n", &seed2));

        *output = new Dataset(ctx, seed, seed2, n, *neighbours, *lengths, *offsets);
    }

private:
    int64 seed_;
    int64 seed2_;
    TF_DISALLOW_COPY_AND_ASSIGN(UniformEdgeDatasetOp);


    class Dataset: public DatasetBase {
    public:
        explicit Dataset(OpKernelContext* ctx, int64 seed, int64 seed2, int32 n,
                         Tensor neighbours, Tensor lengths, Tensor offsets):
            DatasetBase(DatasetContext(ctx)), neighbours_(std::move(neighbours)),
            lengths_(std::move(lengths)), offsets_(std::move(offsets)),
            n_(n), seed_(seed), seed2_(seed2)
        {
        }

        std::unique_ptr<IteratorBase> MakeIteratorInternal(const string& prefix) const override {
            return std::unique_ptr<IteratorBase>(
                new Iterator({this, strings::StrCat(prefix, "::UniformEdgeSample")})
            );
        }

        const DataTypeVector& output_dtypes() const override {
            static DataTypeVector* dtypes = new DataTypeVector({DT_INT32});
            return *dtypes;
        }

        const std::vector<PartialTensorShape>& output_shapes() const override {
            static std::vector<PartialTensorShape>* shapes = new std::vector<PartialTensorShape>({{-1, 2}});
            return *shapes;
        }

        string DebugString() const override {
            return "UniformEdgeDatasetOp::Dataset";
        }
    
    protected:
        Status AsGraphDefInternal(SerializationContext* ctx, DatasetGraphDefBuilder* b, Node** output) const override {
            Node* neighbours = nullptr;
            TF_RETURN_IF_ERROR(b->AddTensor(neighbours_, &neighbours));
            Node* lengths = nullptr;
            TF_RETURN_IF_ERROR(b->AddTensor(lengths_, &lengths));
            Node* offsets = nullptr;
            TF_RETURN_IF_ERROR(b->AddTensor(offsets_, &offsets));

            Node* n = nullptr;
            TF_RETURN_IF_ERROR(b->AddScalar(n_, &n));
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
        int32 n_;
        int64 seed_;
        int64 seed2_;


        class Iterator: public DatasetIterator<Dataset> {
        public:
            explicit Iterator(const Params& params):
                DatasetIterator<Dataset>(params) {
                    generator_.Init(dataset()->seed_, dataset()->seed2_);
                }

            Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors, bool* end_of_sequence) override {
                Tensor out_edges(ctx->allocator({}), DT_INT32, {dataset()->n_, 2});

                uniform_edge_sample(
                    dataset()->n_, generator_,
                    dataset()->neighbours_.vec<int32>(),
                    dataset()->offsets_.vec<int32>(),
                    out_edges.matrix<int32>());
                
                out_tensors->emplace_back(std::move(out_edges));

                return Status::OK();
            }
        private:
            GuardedPhiloxRandom generator_;
        };
    };
};


REGISTER_OP("UniformEdgeDataset")
    .Input("neighbours: int32")
    .Input("offsets: int32")
    .Input("lengths: int32")
    .Input("n: int32")
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


REGISTER_KERNEL_BUILDER(Name("UniformEdgeDataset").Device("CPU"), UniformEdgeDatasetOp);