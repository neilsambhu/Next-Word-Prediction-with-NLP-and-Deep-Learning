$	aTR'���?�f2!ܚ�?_)�Ǻ�?!?�ܵ�?	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?�ܵ�?lxz�,C�?Aŏ1w-�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails_)�Ǻ�?Zd;�O��?AHP�s�b?*	33333�H@2F
Iterator::Modelc�ZB>�?!�/��YI@);�O��n�?1(<	B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!M3�Lns9@)�0�*�?1��+�4@:Preprocessing2U
Iterator::Model::ParallelMapV2�<,Ԛ�}?!�,.B-@)�<,Ԛ�}?1�,.B-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateŏ1w-!?!9�T��u.@){�G�zt?1�ˁ�B
$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�(��0�?!�I�n8�H@)�����g?1aq*?@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��_�Le?!���x�@)��_�Le?1���x�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!?�]�=@)a2U0*�c?1?�]�=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap;�O��n�?!(<	2@)Ǻ���V?1��K�q@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 46.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	u���?;�e�P�?lxz�,C�?!Zd;�O��?	!       "	!       *	!       2$	��W�2ġ?b̀�u�?HP�s�b?!ŏ1w-�?:	!       B	!       J	!       R	!       Z	!       JCPU_ONLYb 