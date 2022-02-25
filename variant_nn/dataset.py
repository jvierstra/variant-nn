
import math
import numpy as np
import h5py
import pysam

import tensorflow as tf

def unpack_genomic_coords(s):
    '''Unpacks genomoic coords string to array
    
        chr1:1-1000 -> [chr1, 0, 1000]
    '''
    chrom, coords = s.split(':')
    start, end = coords.split('-')
    return str(chrom), int(start), int(end)

def onehot_encode_sequence(seq):
    """
    One-hot encode IUPAC DNA sequence with tensors
    - A little 'over-kill' for 1kb sequences, but should scale well
    """
    _embedding_values = np.zeros([90, 4], np.float32)
    _embedding_values[ord('A')] = np.array([1, 0, 0, 0])
    _embedding_values[ord('C')] = np.array([0, 1, 0, 0])
    _embedding_values[ord('G')] = np.array([0, 0, 1, 0])
    _embedding_values[ord('T')] = np.array([0, 0, 0, 1])
    _embedding_values[ord('W')] = np.array([.5, 0, 0, .5])
    _embedding_values[ord('S')] = np.array([0, .5, .5, 0])
    _embedding_values[ord('M')] = np.array([.5, .5, 0, 0])
    _embedding_values[ord('K')] = np.array([0, 0, .5, .5])
    _embedding_values[ord('R')] = np.array([.5, 0, .5, 0])
    _embedding_values[ord('Y')] = np.array([0, .5, 0, .5])
    _embedding_values[ord('B')] = np.array([  0, 1/3, 1/3, 1/3])
    _embedding_values[ord('D')] = np.array([1/3,   0, 1/3, 1/3])
    _embedding_values[ord('H')] = np.array([1/3, 1/3,   0, 1/3])
    _embedding_values[ord('V')] = np.array([1/3, 1/3, 1/3,   0])
    _embedding_values[ord('N')] = np.array([.25, .25, .25, .25])

    embedding_table = tf.Variable(
        _embedding_values,
        name='dna_lookup_table',
        trainable=False)

    with tf.name_scope('dna_onehot_encode'):
        dna_input = tf.io.decode_raw(
            seq, tf.uint8)
        dna_int32 = tf.cast(dna_input, tf.int32)
        onehot_encoded = tf.nn.embedding_lookup(
            embedding_table, dna_int32)
    
    return onehot_encoded

class data_loader(object):
    def __init__(self):
        pass

    def build_dataset(self,
                      batch_size, 
                      shuffle=True,
                      num_to_prefetch=None,
                      num_threads=None,
                      **kwargs):
        raise NotImplementedError()

    def build_generator(self, batch_size=256, shuffle=True, **kwargs):
        raise NotImplementedError()

class h5_data_loader(data_loader):

    def __init__(self,
                 data_files,
                 fasta_file,
                 targets=[],
                 **kwargs):

        # Should check if files exist
        self.data_files = data_files
        self.fasta_file = fasta_file
        self.targets = targets

        assert len(self.data_files) > 0

    def build_dataset(self,
                      batch_size,
                      shuffle=True,
                      num_to_prefetch=5,
                      num_threads=None,
                      **kwargs):
        
        generators = [
            self.build_generator(batch_size=batch_size,
                                 **kwargs)(data_file)
            for data_file in self.data_files]

        def from_generator(x):
            dataset = tf.data.Dataset.from_generator(
                lambda x: generators[x],
                output_signature=self.get_output_signature(),
                args=(x,)
            )
            return dataset

        dataset = tf.data.Dataset.from_tensor_slices(list(range(len(generators))))
        dataset = dataset.interleave(
            lambda x: from_generator(x),
            num_parallel_calls=tf.data.experiental.AUTOTUME if num_threads is None else num_threads
        )

        dataset = dataset.repeat(2)

        if shuffle:
            dataset = dataset.shuffle(10000)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(num_to_prefetch)

        return dataset

    def build_generator(self, 
                        batch_size=256,
                        **kwargs):

        # Class to retreive data from HDF5 file
        class generator(object):
            def __init__(self,
                         fasta_file,
                         batch_size,
                         targets=[]):

                self.fasta_file = fasta_file
                self.batch_size = batch_size
                self.targets = targets

                self.fasta_filehandle = None

            def __call__(self, data_file):

                if not self.fasta_filehandle:
                    self.fasta_filehandle = pysam.FastaFile(self.fasta_file)

                with h5py.File(data_file, mode='w') as h5_filehandle:
                    
                    # Get total records in file and get num batches
                    key = list(h5_filehandle.keys())[0]
                    N = h5_filehandle[key].shape[0]
                    num_batches =int(math.ceil(N/self.batch_size))
                    batch_indicies = list(range(num_batches))

                    for batch_index in batch_indicies:

                        start_index = batch_index*self.batch_size
                        end_index = start_index+self.batch_size
                        if end_index > N:
                            end_index = N

                        slice_metadata, slice_targets = h5_data_loader.get_slice(
                                                h5_filehandle,
                                                start_index,
                                                end_index,
                                                targets=self.targets)
                        
                        onehot_sequence = np.zeros((batch_size, 1000, 4))
                        
                        for i, metadata_string in enumerate(slice_metadata):
                            metadata_dict = dict([
                                kv.split('=') for kv in metadata_string.decode('utf-8').strip().split(';')
                            ])
                            chrom, start, end = unpack_genomic_coords(metadata_dict['features'])
                            seq = self.fasta_filehandle.fetch(chrom, start, end)

                            onehot_sequence[i,:,:] = onehot_encode_sequence(seq)

                        for i in range(self.batch_size):
                            yield (
                                {'sequence': onehot_sequence[i] },
                                slice_targets,
                                1.0)

                self.fasta_filehandle.close()
        
        return generator(self.fasta_file, batch_size, self.targets)
                    
    def get_output_signature(self):

        target_output_signature = {}
        with h5py.open(self.data_files[0]) as h5_filehandle:
            for target_name, target in self.targets:
                key = target[0]
                indicies = target[1]

                if h5_filehandle[key].dtype.char == 'S':
                    dtype = tf.string
                else:
                    dtype = tf.float32

                if not indicies or len(indicies) == 0:
                    shape = (h5_filehandle[key].shape[1:],)
                else:
                    shape = (len(indicies),)
            
                target_output_signature[target_name] = tf.TensorSpec(shape=shape, dtype=dtype)

        return (
            { 'sequence': tf.TensorSpec(shape=(1000, 4), dtype=tf.int32) },
            target_output_signature,
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )

    @staticmethod
    def get_slice(filehandle,
                  start_index,
                  end_index,
                  targets=[]):

        # what datasets to return; also add metadata
        keys = set([v[0] for k, v in targets] + ['metadata'])
        slice = {}

        for key in keys:
            if filehandle[key].dtype.char == 'S':
                slice[key] = filehandle[key][start_index:end_index]
            else:
                key_handle = filehandle[key]
                with key_handle.astype(np.float32):
                    slice[key] = key_handle[start_index:end_index]

        '''
        Filter data points but target sub-indicies
        { target_name: (key, [indicies]) }
        '''
        target_slice = {}
        
        for target_name, target in targets:
            key = target[0]
            indicies = target[1]
            
            if not indicies or len(indicies) == 0:
                target_slice[target_name] = slice[key] # grab everything
            else:
                target_slice[target_name] = slice[key][:, indicies]

        return slice['metadata'],  target_slice