#ifndef RELATIONAL_SGD_SAMPLING_H_INCLUDED
#define RELATIONAL_SGD_SAMPLING_H_INCLUDED

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t index_type;

/*! This struct represents the data for a packed representation of an adjacency list.
 *
 *  This structure describes an adjacency list as a list of neighbours of each vertex in
 * the graph. The set of neighbours of each vertex is given by the section of the neighbours
 * array from offsets[i] to offsets[i] = lengths[i].
 */
typedef struct {
    index_type* neighbours;
    index_type* edge_index;
    index_type* offsets;
    index_type* lengths;
    index_type* vertex_index;

    index_type num_neighbours;
    index_type num_vertex;
} packed_adjacency_list;

typedef packed_adjacency_list* (*alloc_function_type)(void* instance, index_type num_neighbours, index_type num_vertex);

/*! Extracts a p-sample from the given graph from a given vertex sample.
 *
 * This function allocates data for its output. The data is allocated using the
 * passed in function pointer alloc. If NULL, system malloc is used. However,
 * other allocators can be used such as the Python memory allocator.
 *
 * @param[in] graph The original graph to subsample
 * @param[in] candidates A list of subsampled vertices
 * @param[in] num_candidates The length of `candidates`.
 * @param[in] alloc The allocation function to use to allocate the output memory. alloc(num_elem) must
 *                  allocate enough memory to store num_elem elements of type size_t.
 * @param[in] instance This parameter is forwarded to the allocation function.
 */
void p_sample_filter(packed_adjacency_list* graph, index_type* candidates, index_type num_candidates,
                     alloc_function_type alloc, void* instance);


/*! Helper structure used to accelerate the biased random walk sampling.
 */
struct biased_walk_sample_config;

/*! Extracts a biased random walk from the given graph.
 *
 * This function extracts a random walk from the graph according to the given configuration.
 * 
 * @param[in] graph The graph to subsample.
 * @param[in,out] config The configuration for the subsampling. As this stores the random state, it is modified at
 *                       each call to the sampler.
 * @param[out] output_walk An array into which the walk will be written. Must be of length at least that which was
 *                         specified when creating the configuration.
 */
void biased_walk_sample(packed_adjacency_list const* graph, struct biased_walk_sample_config* config, index_type* output_walk);

/*! Initializes and precomputes the structures to accelerate sampling from
 * a graph using biased random walks.
 *
 * @param[in] graph The original graph to subsample.
 * @param[in] p Return parameter.
 * @param[in] q In-out parameter.
 * @param[in] walk_length Length of the walk to sample.
 * @param[in] seed The value to seed the random sequence with.
 * 
 * @returns A pointer to the configuration for this given graph and walk.
 */
struct biased_walk_sample_config* init_biased_walk_config(packed_adjacency_list const* graph, double p, double q, index_type walk_length, uint32_t seed);
void free_biased_walk_config(struct biased_walk_sample_config* config);


/*! Computes the required structure to draw from a multinomial distribution in constant time.
 * 
 * @param[in] num_outcomes The number of possible outcomes for the multinomial.
 * @param[in,out] probabilities An array of length num_outcomes, which contains the probability.
 *                              for each class. After execution, it contains the mixture probabilities
 *                              for the alias samples.
 * @param[out] outcomes An array of length num_outcomes, which contains the paired outcome of the mixture.
 */
void setup_alias_sample(index_type num_outcomes, double* probabilities, index_type* outcomes);
void setup_alias_sample_f(index_type num_outcomes, float* probabilities, index_type* outcomes);

#ifdef __cplusplus
} // extern "C"
#endif

#endif