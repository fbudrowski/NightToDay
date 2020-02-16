# Purpose: testing if the GPUs are working with tf

from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime, timezone
import tensorflow as tf
from read_input import datasets, train_fps_per_dataset, train_ids_per_dataset, test_fps_per_dataset, test_ids_per_dataset, Generator, evaluate_photo

assert(len(tf.config.experimental.list_physical_devices('GPU')) > 0)


start = datetime.now(tz=timezone.utc)


input_generator = Generator(test_fps_per_dataset, train_fps_per_dataset, datasets, False)

stop1 = datetime.now(tz=timezone.utc)

image = input_generator.get_samples(1, 0)

stop2 = datetime.now(tz=timezone.utc)

print("Generator initiated in {}s".format((stop1 - start).total_seconds()))
print("Image generated in {}s".format((stop2 - stop1).total_seconds()))