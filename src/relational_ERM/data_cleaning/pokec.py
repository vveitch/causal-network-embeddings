"""
Load and process pokec social network

Data includes:
edge structure
user profiles

data from https://snap.stanford.edu/data/soc-Pokec.html
"""

import os
import argparse
from collections import namedtuple

import numpy as np
import pandas as pd
import tensorflow as tf

from relational_ERM.graph_ops.representations import PackedAdjacencyList


def _load_profiles(profile_file):
    # df = pd.read_csv('filename.tar.gz', compression='gzip', header=0, sep=',', quotechar='"'

    names = str.split("user_id public completion_percentage gender region last_login registration age body "
                      "I_am_working_in_field spoken_languages hobbies I_most_enjoy_good_food pets body_type "
                      "my_eyesight eye_color hair_color hair_type completed_level_of_education favourite_color "
                      "relation_to_smoking relation_to_alcohol sign_in_zodiac on_pokec_i_am_looking_for love_is_for_me "
                      "relation_to_casual_sex my_partner_should_be marital_status children relation_to_children "
                      "I_like_movies I_like_watching_movie I_like_music I_mostly_like_listening_to_music "
                      "the_idea_of_good_evening I_like_specialties_from_kitchen fun I_am_going_to_concerts "
                      "my_active_sports my_passive_sports profession I_like_books life_style music cars politics "
                      "relationships art_culture hobbies_interests science_technologies computers_internet education "
                      "sport movies travelling health companies_brands more")

    usecols = str.split("user_id public completion_percentage gender region last_login registration age")

    # extracols = str.split("spoken_languages " # half nan
    #                       "body body_type " # both half nan
    #                       "eye_color hair_color hair_type " # all appx half nan
    #                       "completed_level_of_education" # appx half nan
    #                       "favourite_color " # half nan
    #                       "relation_to_smoking relation_to_alcohol " # half nan
    #                       "sign_in_zodiac " # half nan
    #                       "relation_to_casual_sex marital_status children " # 0.62, 0.54, 0.7 nan
    #                       "I_like_movies I_like_watching_movie " # both half... but meh
    #                       "I_like_music I_mostly_like_listening_to_music " # 0.53, 0.54
    #                       "I_like_books " # 0.59
    #                       )

    extracols = str.split("completed_level_of_education " # appx half nan
                          "sign_in_zodiac " # half nan
                          "relation_to_casual_sex " # 0.62
                          "I_like_books " # 0.59
                          )


    usecols = usecols + extracols

    profiles = pd.read_csv(profile_file, names=names, index_col=False, usecols=usecols, compression='gzip', header=None, sep='\t')
    profiles.set_index('user_id', inplace=True, drop=False)
    return profiles


def _process_profiles(profiles):
    """
    # Fix datatypes
    """
    # keep_attributes = str.split("user_id public completion_percentage gender region last_login registration age")
    # p2=profiles[keep_attributes]
    p2 = profiles
    p2['region'] = p2['region'].astype('category')
    p2['public'] = p2['public'].astype('category')
    p2['gender'] = p2['gender'].astype('category')
    p2.loc[p2.age == 0, 'age'] = np.nan

    p2['completed_level_of_education'] = p2['completed_level_of_education'].isna()
    p2['sign_in_zodiac'] = p2['sign_in_zodiac'].isna()
    p2['relation_to_casual_sex'] = p2['relation_to_casual_sex'].isna()
    p2['I_like_books'] = p2['I_like_books'].isna()

    p2['last_login'] = pd.to_datetime(p2['last_login'])
    p2['registration'] = pd.to_datetime(p2['registration'])

    return p2


def preprocess_data(data_directory='../data/pokec'):
    link_file = os.path.join(data_directory, 'soc-pokec-relationships.txt.gz')
    profile_file = os.path.join(data_directory, 'soc-pokec-profiles.txt.gz')

    edge_list = np.loadtxt(link_file, dtype=np.int32)
    profiles = _load_profiles(profile_file)
    profiles = _process_profiles(profiles)

    # relational ERM code expects 0-indexed vertices, but data is 1-indexed
    profiles['user_id'] = profiles['user_id'] - 1
    edge_list = edge_list - 1

    return {
        'edge_list': edge_list,
        'profiles': profiles
    }


def preprocess_packed_adjacency_list(edge_list):
    from ..graph_ops.representations import create_packed_adjacency_from_redundant_edge_list

    # Load the current edge list, and go to canonical form
    # i.e., remove self-edges and convert to undirected graph
    edge_list = edge_list[edge_list[:, 0] != edge_list[:, 1], :]
    edge_list.sort(axis=-1)
    edge_list = np.unique(edge_list, axis=0)

    # Compute redundant edge list
    edge_list = np.concatenate((edge_list, np.flip(edge_list, axis=1)))
    adjacency_list = create_packed_adjacency_from_redundant_edge_list(edge_list)

    return {
        'neighbours': adjacency_list.neighbours,
        'offsets': adjacency_list.offsets,
        'lengths': adjacency_list.lengths,
        'vertex_index': adjacency_list.vertex_index
    }


def _edges_in_region(edge_list, vertices_in_region):
    edge_list = np.copy(edge_list)
    edge_in_region = np.isin(edge_list[:, 0], vertices_in_region)
    edge_list = edge_list[edge_in_region]
    edge_in_region = np.isin(edge_list[:, 1], vertices_in_region)
    edge_list = edge_list[edge_in_region]
    return edge_list.shape[0]


def subset_to_region(edge_list, profiles, regions=None):
    """
    subset to particular (geographical) region
    """
    if regions is None:
        #
        regions = ['zilinsky kraj, zilina', 'zilinsky kraj, cadca', 'zilinsky kraj, namestovo']

    user_in_region = np.zeros_like(profiles['region'], dtype=np.bool)
    for region in regions:
        print((profiles['region'] == region).sum())
        user_in_region = np.logical_or(user_in_region,
                                       profiles['region'] == region)

    # regions = profiles.region.cat.categories
    # for candidate_region in regions:
    #     if region in candidate_region:
    #         print(candidate_region)
    #         user_in_region = (profiles['region'] == candidate_region)
    #         print(user_in_region.sum())
    #         vertices_in_region = profiles.loc[user_in_region]['user_id'].values
    #         print(_edges_in_region(edge_list, vertices_in_region) / user_in_region.sum())

    vertices_in_region = profiles.loc[user_in_region]['user_id'].values

    edge_list = np.copy(edge_list)
    edge_in_region = np.isin(edge_list[:, 0], vertices_in_region)
    edge_list = edge_list[edge_in_region]
    edge_in_region = np.isin(edge_list[:, 1], vertices_in_region)
    edge_list = edge_list[edge_in_region]

    # some users may be isolates in the new graph, and must be purged
    present_user_ids = np.unique(edge_list)
    present_user_indicator = np.isin(profiles['user_id'].values, present_user_ids)
    regional_profiles = profiles[present_user_indicator]

    regional_profiles.set_index('user_id')

    # reindex to make subgraph contiguous
    # specifically, relabel the edges so that the vertex label is that users index in profiles
    index_to_user_id = regional_profiles['user_id'].values
    user_id_to_index = np.zeros(np.max(index_to_user_id)+1, dtype=np.int32)-1
    user_id_to_index[index_to_user_id] = np.arange(index_to_user_id.shape[0])

    edge_list = user_id_to_index[edge_list]

    regional_profiles.to_pickle("regional_profiles.pkl")
    np.savez_compressed('regional_pokec_links.npz', edge_list=edge_list)
    packed_adjacency_list_data = preprocess_packed_adjacency_list(edge_list)
    np.savez_compressed('regional_pokec_links_processed.npz', **packed_adjacency_list_data)


GraphData = namedtuple('GraphData', ['edge_list',
                                     'weights',
                                     'adjacency_list',
                                     'num_vertices'])


def load_data_pokec(data_folder=None):
    """ Loads pre-processed pokec data

    Parameters
    ----------
    data_folder: The path to the pre-processed data

    Returns
    -------
    An instance of GraphDataPokec containing the parsed graph data for the dataset.
    """
    if data_folder is None:
        data_folder = '../data/networks/pokec'

    # data_folder = '../data/blog_catalog_3/blog_catalog.npz'
    # data_folder = '../dat/wikipedia_word_coocurr/wiki_pos.npz'

    # use tensorflow loading to support loading from
    # cloud providers
    data_path = os.path.join(data_folder, 'pokec_links.npz')
    with tf.gfile.Open(data_path, mode='rb') as f:
        loaded = np.load(f, allow_pickle=False)

    edge_list = loaded['edge_list'].astype(np.int32)

    if 'weights' in loaded:
        weights = loaded['weights'].astype(np.float32)
    else:
        weights = np.ones(edge_list.shape[0], dtype=np.float32)

    # load pre-processed adjacency list, because the operation is very slow
    data_path = os.path.join(data_folder, 'pokec_links_processed.npz')
    with tf.gfile.Open(data_path, mode='rb') as f:
        loaded = np.load(f, allow_pickle=False)
    neighbours = loaded['neighbours']
    offsets = loaded['offsets']
    lengths = loaded['lengths']
    vertex_index = loaded['vertex_index']

    adjacency_list = PackedAdjacencyList(neighbours, weights, offsets, lengths, vertex_index)

    graph_data = GraphData(edge_list=edge_list,
                           weights=weights,
                           adjacency_list=adjacency_list,
                           num_vertices=len(vertex_index))

    profile_file = os.path.join(data_folder, 'profiles.pkl')
    profiles = pd.read_pickle(profile_file)

    return graph_data, profiles

#
# def add_parser_sampling_arguments(parser=None):
#     if parser is None:
#         parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--batch-size', type=int, default=None, help='minibatch size')
#     parser.add_argument('--dataset-shards', type=int, default=None, help='dataset parallelism')
#
#     parser.add_argument('--num-edges', type=int, default=800,
#                         help='Number of edges per sample.')
#
#     parser.add_argument('--window-size', type=int, default=1,
#                         help='Context size for optimization. Default is 10.')
#
#     parser.add_argument('--num-negative', type=int, default=0,
#                         help='negative examples per vertex for negative sampling')
#
#     parser.add_argument('--num-negative-total', type=int, default=None,
#                         help='total number of negative vertices sampled')
#
#     return parser
#
#
# def subset_by_random_walk(graph_data, profiles, args):
#     dataset_fn_train = make_dataset('biased-walk', args)
#     dataset = dataset_fn_train(graph_data, 0)
#     itr = dataset.make_one_shot_iterator()
#     sample = itr.get_next()
#     subset_edgelist = sample['edge_list']
#
#     vertices_in_subset = np.unique(subset_edgelist)
#     profiles_in_subset = profiles[vertices_in_subset]
#
#     return sample


def _standardize(x):
    return (x-x.mean())/x.std()


def process_pokec_attributes(profiles):
    """
    Final processing on pokec attributes for tensorflow compatibility
    """

    included_features = str.split("public "
                                  "completion_percentage "
                                  "gender "
                                  "region "
                                  "age "
                                  "completed_level_of_education "
                                  "sign_in_zodiac "
                                  "relation_to_casual_sex "
                                  "I_like_books "
                                  "recent_login "
                                  "old_school "
                                  "scaled_registration "
                                  "scaled_age "
                                  )

    profiles['recent_login'] = (profiles['last_login'] < pd.Timestamp(2012, 5, 1))
    profiles['old_school'] = (profiles['registration'] < pd.Timestamp(2009, 1, 1))

    profiles['scaled_registration'] = (profiles['registration'] - profiles['registration'].min()) / pd.offsets.Day(1)
    profiles['scaled_registration'] = _standardize(profiles['scaled_registration'])

    profiles['scaled_age'] = _standardize(profiles['age'])

    # preprocess to tensorflow compatible formats
    # convert categorical to int32
    cat_columns = profiles.select_dtypes(['category']).columns
    profiles[cat_columns] = profiles[cat_columns].apply(lambda x: x.cat.codes)
    profiles[cat_columns] = profiles[cat_columns].apply(lambda x: x.astype(np.int32))
    bool_columns = profiles.select_dtypes([bool]).columns
    profiles[bool_columns] = profiles[bool_columns].apply(lambda x: x.astype(np.int32))

    profiles['age'][profiles['age'].isna()] = -1.

    profiles['age'] = profiles['age'].astype(np.float32)
    profiles['completion_percentage'] = profiles['completion_percentage'].astype(np.float32)

    pokec_features = {}
    for feature in included_features:
        pokec_features[feature] = profiles[feature].values

    return pokec_features


def main():
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    if args.data_dir is not None:
        data = preprocess_data(args.data_dir)
    else:
        data = preprocess_data()

    data['profiles'].to_pickle("profiles.pkl")
    np.savez_compressed('pokec_links.npz', edge_list=data['edge_list'])

    packed_adjacency_list_data = preprocess_packed_adjacency_list(data['edge_list'])
    np.savez_compressed('pokec_links_processed.npz', **packed_adjacency_list_data)

    data_folder = os.getcwd()

    graph_data, profiles = load_data_pokec(data_folder)
    subset_to_region(graph_data.edge_list, profiles, regions=None)


if __name__ == '__main__':
    main()

