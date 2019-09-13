#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/dataset.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/util/guarded_philox_random.h>
#include <tensorflow/core/lib/random/random_distributions.h>

using namespace tensorflow;


/*! Uniform random walk dataset. */
class RandomWalkDatasetOp : public DatasetOpKernel {
public:
    explicit RandomWalkDatasetOp(OpKernelConstruction* ctx): DatasetOpKernel(ctx) {
    }

    void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
        const Tensor* neighbours;
        const Tensor* lengths;
        const Tensor* offsets;

        int32 walk_length;

        int64 seed;
        int64 seed2;

        OP_REQUIRES_OK(ctx, ParseScalarArgument<int32>(ctx, "walk_length", &walk_length));
        OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed", &seed));
        OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed2", &seed2));

        OP_REQUIRES_OK(ctx, ctx->input("neighbours", &neighbours));
        OP_REQUIRES_OK(ctx, ctx->input("lengths", &lengths));
        OP_REQUIRES_OK(ctx, ctx->input("offsets", &offsets));


        *output = new Dataset(ctx, walk_length, seed, seed2, *neighbours, *lengths, *offsets);
    }
private:
    TF_DISALLOW_COPY_AND_ASSIGN(RandomWalkDatasetOp);

    class Dataset: public DatasetBase {
    public:
        explicit Dataset(OpKernelContext* ctx, int32 walk_length, int64 seed, int64 seed2,
                         Tensor neighbours, Tensor lengths, Tensor offsets)
            : DatasetBase(DatasetContext(ctx)), neighbours_(std::move(neighbours)), lengths_(std::move(lengths)),
                offsets_(std::move(offsets)), shapes_({{walk_length}}), walk_length_(walk_length),
                seed_(seed), seed2_(seed2)
            {}


        std::unique_ptr<IteratorBase> MakeIteratorInternal(const string& prefix) const override {
            return std::unique_ptr<IteratorBase>(
                new Iterator({this, strings::StrCat(prefix, "::RandomWalkSample")})
            );
        }

        const DataTypeVector& output_dtypes() const override {
            static DataTypeVector* dtypes = new DataTypeVector({DT_INT32});
            return *dtypes;
        }

        const std::vector<PartialTensorShape>& output_shapes() const override {
            return shapes_;
        }

        string DebugString() const override {
            return "RandomWalkDatasetOp::Dataset";
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
        const int32 walk_length_;
        const int64 seed_;
        const int64 seed2_;
        const std::vector<PartialTensorShape> shapes_;

        class Iterator: public DatasetIterator<Dataset> {
        public:
            explicit Iterator(const Params& params):
                DatasetIterator<Dataset>(params), generator_(dataset()->seed_, dataset()->seed2_)
            {}

            Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors, bool* end_of_sequence) override {
                auto walk_length = dataset()->walk_length_;
                Tensor out_walk(ctx->allocator({}), {DT_INT32}, {walk_length});

                auto out_walk_vec = out_walk.vec<int32>();
                const auto kResultElementCount = random::PhiloxRandom::kResultElementCount;

                {
                    mutex_lock lock(mu_);

                    const auto num_full_samples = walk_length / kResultElementCount;

                    for(int i = 0; i < num_full_samples; ++i) {
                        auto samples = generator_();
                        std::copy(&samples[0], &samples[0] + kResultElementCount, out_walk_vec.data() + kResultElementCount * i);
                    }

                    auto samples_end = generator_();
                    const auto remaining = walk_length % kResultElementCount;
                    std::copy(
                        &samples_end[0], &samples_end[0] + remaining,
                        out_walk_vec.data() + num_full_samples * kResultElementCount);

                    num_random_samples_ += num_full_samples + 1;
                }

                auto neighbours_vec = dataset()->neighbours_.vec<int32>();
                auto lengths_vec = dataset()->lengths_.vec<int32>();
                auto offsets_vec = dataset()->offsets_.vec<int32>();

                int32 current_vertex = static_cast<uint32>(out_walk_vec(0)) % lengths_vec.dimension(0);
                out_walk_vec(0) = current_vertex;

                for(int i = 1; i < walk_length; ++i) {
                    auto neighbours_offset = static_cast<uint32>(out_walk_vec(i)) % lengths_vec(current_vertex);
                    current_vertex = neighbours_vec(offsets_vec(current_vertex) + neighbours_offset);
                    out_walk_vec(i) = current_vertex;
                }

                out_tensors->emplace_back(std::move(out_walk));

                return Status::OK();
            }
        private:
            mutex mu_;
            random::PhiloxRandom generator_ GUARDED_BY(mu_);
            int64 num_random_samples_ GUARDED_BY(mu_);
        };
    };
};


REGISTER_OP("RandomWalkDataset")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Input("neighbours: int32")
    .Input("offsets: int32")
    .Input("lengths: int32")
    .Input("walk_length: int32")
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

        return shape_inference::ScalarShape(c);
    });

REGISTER_KERNEL_BUILDER(Name("RandomWalkDataset").Device("CPU"), RandomWalkDatasetOp);