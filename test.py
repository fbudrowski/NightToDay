# Purpose: testing if the GPUs are working with tf

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

assert(len(tf.config.experimental.list_physical_devices('GPU')) > 0)