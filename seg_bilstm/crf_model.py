#!/usr/bin/python
# -*- coding:utf-8 -*-
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.framework import dtypes

class CrfDecodeForwardRnnCell(rnn_cell.RNNCell):
    """Computes the forward decoding in a linear-chain CRF.
    """

    def __init__(self, transition_params):
        """Initialize the CrfDecodeForwardRnnCell.
        Args:
          transition_params: A [num_tags, num_tags] matrix of binary
            potentials. This matrix is expanded into a
            [1, num_tags, num_tags] in preparation for the broadcast
            summation occurring within the cell.
        """
        self._transition_params = array_ops.expand_dims(transition_params, 0)
        self._num_tags = transition_params.get_shape()[0].value

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None):
        """Build the CrfDecodeForwardRnnCell.
        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous step's
                score values.
          scope: Unused variable scope of this cell.
        Returns:
          backpointers: [batch_size, num_tags], containing backpointers.
          new_state: [batch_size, num_tags], containing new score values.
        """
        # For simplicity, in shape comments, denote:
        # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
        state = array_ops.expand_dims(state, 2)  # [B, O, 1]

        # This addition op broadcasts self._transitions_params along the zeroth
        # dimension and state along the second dimension.
        # [B, O, 1] + [1, O, O] -> [B, O, O]
        transition_scores = state + self._transition_params  # [B, O, O]
        new_state = inputs + math_ops.reduce_max(transition_scores, [1])  # [B, O]
        backpointers = math_ops.argmax(transition_scores, 1)
        backpointers = math_ops.cast(backpointers, dtype=dtypes.int32)  # [B, O]
        return backpointers, new_state


class CrfDecodeBackwardRnnCell(rnn_cell.RNNCell):
    """Computes backward decoding in a linear-chain CRF.
    """

    def __init__(self, num_tags):
        """Initialize the CrfDecodeBackwardRnnCell.
        Args:
          num_tags
        """
        self._num_tags = num_tags

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 1

    def __call__(self, inputs, state, scope=None):
        """Build the CrfDecodeBackwardRnnCell.
        Args:
          inputs: [batch_size, num_tags], backpointer of next step (in time order).
          state: [batch_size, 1], next position's tag index.
          scope: Unused variable scope of this cell.
        Returns:
          new_tags, new_tags: A pair of [batch_size, num_tags]
            tensors containing the new tag indices.
        """
        state = array_ops.squeeze(state, axis=[1])  # [B]
        batch_size = array_ops.shape(inputs)[0]
        b_indices = math_ops.range(batch_size)  # [B]
        indices = array_ops.stack([b_indices, state], axis=1)  # [B, 2]
        new_tags = array_ops.expand_dims(
            gen_array_ops.gather_nd(inputs, indices),  # [B]
            axis=-1)  # [B, 1]

        return new_tags, new_tags

def crf_decode(potentials, transition_params, sequence_length):
    """Decode the highest scoring sequence of tags in TensorFlow.
    This is a function for tensor.
    Args:
      potentials: A [batch_size, max_seq_len, num_tags] tensor, matrix of
                unary potentials.
      transition_params: A [num_tags, num_tags] tensor, matrix of
                binary potentials.
      sequence_length: A [batch_size] tensor, containing sequence lengths.
    Returns:
      decode_tags: A [batch_size, max_seq_len] tensor, with dtype tf.int32.
                  Contains the highest scoring tag indicies.
      best_score: A [batch_size] tensor, containing the score of decode_tags.
    """
    # For simplicity, in shape comments, denote:
    # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
    num_tags = potentials.get_shape()[2].value

    # Computes forward decoding. Get last score and backpointers.
    crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params)
    initial_state = array_ops.slice(potentials, [0, 0, 0], [-1, 1, -1])
    initial_state = array_ops.squeeze(initial_state, axis=[1])  # [B, O]
    inputs = array_ops.slice(potentials, [0, 1, 0], [-1, -1, -1])  # [B, T-1, O]
    backpointers, last_score = rnn.dynamic_rnn(
        crf_fwd_cell,
        inputs=inputs,
        sequence_length=sequence_length - 1,
        initial_state=initial_state,
        time_major=False,
        dtype=dtypes.int32)  # [B, T - 1, O], [B, O]
    backpointers = gen_array_ops.reverse_sequence(
        backpointers, sequence_length - 1, seq_dim=1)  # [B, T-1, O]

    # Computes backward decoding. Extract tag indices from backpointers.
    crf_bwd_cell = CrfDecodeBackwardRnnCell(num_tags)
    initial_state = math_ops.cast(math_ops.argmax(last_score, axis=1),
                                  dtype=dtypes.int32)  # [B]
    initial_state = array_ops.expand_dims(initial_state, axis=-1)  # [B, 1]
    decode_tags, _ = rnn.dynamic_rnn(
        crf_bwd_cell,
        inputs=backpointers,
        sequence_length=sequence_length - 1,
        initial_state=initial_state,
        time_major=False,
        dtype=dtypes.int32)  # [B, T - 1, 1]
    decode_tags = array_ops.squeeze(decode_tags, axis=[2])  # [B, T - 1]
    decode_tags = array_ops.concat([initial_state, decode_tags], axis=1)  # [B, T]
    decode_tags = gen_array_ops.reverse_sequence(
        decode_tags, sequence_length, seq_dim=1)  # [B, T]

    best_score = math_ops.reduce_max(last_score, axis=1)  # [B]
    return decode_tags, best_score

