from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import tensorflow as tf
import os


def _append_to_key(dict, key, values):
    if key in dict:
        existing_values = dict[key]
    else:
        existing_values = []

    existing_values.extend(values)
    dict[key] = existing_values
    return dict


class TensorflowExtension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        # TODO: compatibility with tensorflow <= 1.7.0 where we need to add nsync directory
        _append_to_key(kwargs, 'include_dirs', [tf.sysconfig.get_include()])

        if os.name == 'nt':
            _append_to_key(kwargs, 'library_dirs', [os.path.join(tf.sysconfig.get_lib(), 'python')])
            _append_to_key(kwargs, 'libraries', ['_pywrap_tensorflow_internal'])
            _append_to_key(kwargs, 'extra_compile_args',
                           ['/DCOMPILER_MSVC', '/DNOMINMAX', '/DWIN32_LEAN_AND_MEAN', '/DVC_EXTRALEAN',
                            '/wd4267', '/wd4244', '/permissive-'])
        else:
            _append_to_key(kwargs, 'library_dirs', [tf.sysconfig.get_lib()])
            _append_to_key(kwargs, 'libraries', ['tensorflow_framework'])
            _append_to_key(kwargs, 'extra_compile_args',
                           [f for f in tf.sysconfig.get_compile_flags()
                            if not f.startswith('-I')] +
                           ['-std=c++11'])

        super().__init__(name, sources, *args, **kwargs)


class BuildExtCustom(build_ext):
    def get_export_symbols(self, ext):
        if isinstance(ext, TensorflowExtension):
            return ext.export_symbols

        return super().get_export_symbols(ext)


extensions_tensorflow = [
    TensorflowExtension('relational_ERM.tensorflow_ops._datasets_tensorflow',
                        ['relational_ERM/tensorflow_ops/biased_walk_dataset.cpp',
                         'relational_ERM/tensorflow_ops/p_sampling_dataset.cpp',
                         'relational_ERM/tensorflow_ops/uniform_edge_dataset.cpp',
                         'relational_ERM/tensorflow_ops/random_walk_dataset.cpp']),
    TensorflowExtension('relational_ERM.tensorflow_ops._adapters_tensorflow',
                        ['relational_ERM/tensorflow_ops/adjacency_to_edge_list.cpp',
                         'relational_ERM/tensorflow_ops/induced_subgraph.cpp',
                         'relational_ERM/tensorflow_ops/induced_ego_sample.cpp',
                         'relational_ERM/tensorflow_ops/concatenate_slices.cpp']),
    TensorflowExtension('relational_ERM.tensorflow_ops._array_ops_tensorflow',
                        ['relational_ERM/tensorflow_ops/concatenate_slices.cpp',
                         'relational_ERM/tensorflow_ops/packed_to_sparse.cpp',
                         'relational_ERM/tensorflow_ops/repeat.cpp',
                         'relational_ERM/tensorflow_ops/batch_length_to_segment.cpp'])
]

setup(
    name='relational_ERM',
    version='0.1.0',
    author='Victor Veitch and Wenda Zhou',
    author_email='wz2335@columbia.edu',
    packages=find_packages(),
    ext_modules=extensions_tensorflow,
    cmdclass={'build_ext': BuildExtCustom},
    install_requires=[
        'numpy>=1.13'
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.8.0'],
        'tensorflow-gpu': ['tensorflow-gpu>=1.8.0']
    }
)
