# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Writes out text data as tfrecords that ELECTRA can be pre-trained on."""

import argparse
import multiprocessing
import os
import random
import time
import sys
import tensorflow.compat.v1 as tf

from model import tokenization
from util import utils
from util.gen import replace_sentence


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


class ExampleBuilder(object):
  """Given a stream of input text, creates pretraining examples."""

  def __init__(self, tokenizer, max_length):
    self._tokenizer = tokenizer
    self._current_sentences = []
    self._original_sentences = []
    self._label = []
    self._current_length = 0
    self._max_length = max_length
    self._target_length = max_length

  def add_line(self, original_line, line):
    """Adds a line of text to the current example being built."""
    line = line.strip().replace("\n", " ")
    original_line = original_line.strip().replace("\n", " ")
    
    labelList = list() 
    originalList = original_line.split(" ")
    replacedList = line.split(" ")
    if (len(originalList) != len(replacedList)):
        print("Warning: originalList and replaceList have different number of words")
        print(f"originalList: {originalList}\nreplacedList: {replacedList}")
    
    if (not line) and self._current_length != 0:  # empty lines separate docs
        return self._create_example()
    bert_tokens = self._tokenizer.tokenize(line)
    original_tokens = self._tokenizer.tokenize(original_line)

    for i in range(len(replacedList)):
        word = replacedList[i]
        tempToken = self._tokenizer.tokenize(word)
        if (len(originalList) < i):
          labeling = 1
        else:
          if (replacedList[i] == originalList[i]):
            labeling = 0
          else:
            labeling = 1
        for i in range(len(tempToken)):
            labelList.append(labeling)
    assert (len(labelList) == len(bert_tokens))

    bert_tokids = self._tokenizer.convert_tokens_to_ids(bert_tokens)
    original_tokids = self._tokenizer.convert_tokens_to_ids(original_tokens)
    self._current_sentences.append(bert_tokids)
    self._original_sentences.append(original_tokids)
    self._label.append(labelList)
    self._current_length += len(bert_tokids)
    if self._current_length >= self._target_length:
      return self._create_example()
    return None

  def _create_example(self):
    """Creates a pre-training example from the current list of sentences."""
    # small chance to only have one segment as in classification tasks
    if random.random() < 0.1:
      first_segment_target_length = 100000
    else:
      # -3 due to not yet having [CLS]/[SEP] tokens in the input text
      first_segment_target_length = (self._target_length - 3) // 2

    first_segment = []
    first_label = []
    for sentence, label in zip(self._current_sentences, self._label):
        first_segment += sentence
        first_label += label

    # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
    first_segment = first_segment[:self._max_length - 2] 
    label = label[:self._max_length-2]

    # prepare to start building the next example
    self._current_sentences = []
    self._label = []
    self._current_length = 0
    # small chance for random-length instead of max_length-length example
    if random.random() < 0.05:
      self._target_length = random.randint(5, self._max_length)
    else:
      self._target_length = self._max_length

    return self._make_tf_example(first_segment, first_label)

  def _make_tf_example(self, first_segment, first_label):
    vocab = self._tokenizer.vocab
    input_ids = [vocab["[CLS]"]] + first_segment + [vocab["[SEP]"]]
    input_mask = [1] * len(input_ids)
    input_ids += [0] * (self._max_length - len(input_ids))
    input_mask += [0] * (self._max_length - len(input_mask))

    # add 0 to first and last part of self._label do to [CLS] and [SEP] token
    label = [0] + first_label + [0] 
    label += [0] * (self._max_length - len(label))
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "input_ids": create_int_feature(input_ids),
        "input_mask": create_int_feature(input_mask),
        "label": create_int_feature(label), 
    }))
    return tf_example


class ExampleWriter(object):
  """Writes pre-training examples to disk."""

  def __init__(self, job_id, vocab_file, output_dir, max_seq_length,
               num_jobs, blanks_separate_docs, do_lower_case,
               num_out_files=1000, random = random):
    self._blanks_separate_docs = blanks_separate_docs
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=do_lower_case)

    self._example_builder = ExampleBuilder(tokenizer, max_seq_length)
    self.random = random
    self._writers = []
    for i in range(num_out_files):
      if i % num_jobs == job_id:
        output_fname = os.path.join(
            output_dir, "pretrain_data.tfrecord-{:}-of-{:}".format(
                i, num_out_files))
        self._writers.append(tf.io.TFRecordWriter(output_fname))
    self.n_written = 0

  def write_fake_examples(self, line, total_tokens):
    lineList = line.split("\n")
    retVal = ""
    for i in range(len(lineList)):
       curLine = lineList[i]
       replaceNum = int(len(curLine.split(" ")) * 0.15)
#       print("[Original] {}".format(curLine))
       lineList[i] = replace_sentence(curLine, replaceNum, total_tokens, self.random) 
#       print("[Replaced] {}".format(lineList[i]))
       retVal += lineList[i]
    return retVal

  def write_examples(self, input_file, total_tokens):
    """Writes out examples from the provided input file."""
    with tf.io.gfile.GFile(input_file) as f:
      for line in f:
        """
        original_line: line from the dataset
        line         : replaced original_line
        """
        original_line = line.strip() 
        if original_line or self._blanks_separate_docs:
          replaced_line = self.write_fake_examples(original_line, total_tokens)
          example = self._example_builder.add_line(original_line, replaced_line)
          if example:
            self._writers[self.n_written % len(self._writers)].write(
                example.SerializeToString())
            self.n_written += 1
      example = self._example_builder.add_line("", "")
      if example:
        self._writers[self.n_written % len(self._writers)].write(
            example.SerializeToString())
        self.n_written += 1

  def finish(self):
    for writer in self._writers:
      writer.close()


def write_examples(job_id, args):
  """A single process creating and writing out pre-processed examples."""

  def log(*args):
    msg = " ".join(map(str, args))
    print("Job {}:".format(job_id), msg)

  log("Creating example writer")
  example_writer = ExampleWriter(
    job_id=job_id,
    vocab_file=args.vocab_file,
    output_dir=args.output_dir,
    max_seq_length=args.max_seq_length,
    num_jobs=args.num_processes,
    blanks_separate_docs=args.blanks_separate_docs,
    do_lower_case=args.do_lower_case,
    random=args.random
  )
  log("Writing tf examples")
  fnames = sorted(tf.io.gfile.listdir(args.corpus_dir))
  fnames = [f for (i, f) in enumerate(fnames)
            if i % args.num_processes == job_id]
  random.shuffle(fnames)

  start_time = time.time()
  for file_no, fname in enumerate(fnames):
    if file_no > 0:
      elapsed = time.time() - start_time
      log("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s, ETA: {:}s, "
          "{:} examples written".format(
              file_no, len(fnames), 100.0 * file_no / len(fnames), int(elapsed),
              int((len(fnames) - file_no) / (file_no / elapsed)),
              example_writer.n_written))
    example_writer.write_examples(os.path.join(args.corpus_dir, fname), total_tokens)
  example_writer.finish()
  log("Done!")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--corpus-dir", required=True,
                      help="Location of pre-training text files.")
  parser.add_argument("--vocab-file", required=True,
                      help="Location of vocabulary file.")
  parser.add_argument("--output-dir", required=True,
                      help="Where to write out the tfrecords.")
  parser.add_argument("--max-seq-length", default=128, type=int,
                      help="Number of tokens per example.")
  parser.add_argument("--num-processes", default=1, type=int,
                      help="Parallelize across multiple processes.")
  parser.add_argument("--blanks-separate-docs", default=True, type=bool,
                      help="Whether blank lines indicate document boundaries.")
  parser.add_argument("--do-lower-case", dest='do_lower_case',
                      action='store_true', help="Lower case input text.")
  parser.add_argument("--no-lower-case", dest='do_lower_case',
                      action='store_false', help="Don't lower case input text.")
  parser.set_defaults(do_lower_case=True)
  args = parser.parse_args()

  utils.rmkdir(args.output_dir)
  if args.num_processes == 1:
    write_examples(0, args)
  else:
    jobs = []
    for i in range(args.num_processes):
      job = multiprocessing.Process(target=write_examples, args=(i, args))
      jobs.append(job)
      job.start()
    for job in jobs:
      job.join()


if __name__=="__main__":
   main()
