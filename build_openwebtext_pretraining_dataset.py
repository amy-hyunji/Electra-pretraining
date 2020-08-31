import argparse
import multiprocessing
import random
import tarfile
import time
import os
import tensorflow.compat.v1 as tf
import sys

import build_pretraining_dataset
from util import utils, gen

def write_examples(job_id, fnames, owt_dir, args):
	if args.random:
		job_tmp_dir = os.path.join("./dataset/", f"random{args.epoch}", "job_"+str(job_id))
	else:
		job_tmp_dir = os.path.join("./dataset/", f"wordnet{args.epoch}", "job_"+str(job_id))

	def log(*args):
		msg = " ".join(map(str, args))
		print("Job {}:".format(job_id), msg)

	log(f"directory: {job_tmp_dir}")

	log("Creating example writer")
	example_writer = build_pretraining_dataset.ExampleWriter(
			job_id = job_id,
			vocab_file = os.path.join("./dataset/", "vocab.txt"),
			output_dir = os.path.join("./dataset/", f"random{args.epoch}_tfrecords") if args.random else os.path.join("./dataset/", f"wordnet{args.epoch}_tfrecords"),
			max_seq_length = args.max_seq_length,
			num_jobs = args.num_processes,
			blanks_separate_docs = False,
			do_lower_case = args.do_lower_case,
			random = args.random,
			)
	total_tokens = []

	for file_no, fname in enumerate(fnames):
		if (file_no % 1000 == 0):
			log(f"{file_no}/{len(fnames)} ({round(float(file_no)/float(len(fnames))*100, 3)}%)")
		utils.rmkdir(job_tmp_dir)
		with tarfile.open(os.path.join(owt_dir, fname)) as f:
			f.extractall(job_tmp_dir)
			f.close()
		extracted_files = tf.io.gfile.listdir(job_tmp_dir)
		random.shuffle(extracted_files)
		for i, txt_fnames in enumerate(extracted_files):
			for iter in range(5):
				example_writer.write_examples(os.path.join(job_tmp_dir, txt_fnames), total_tokens)
	example_writer.finish()
	log("Done!")


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-dir", default="./dataset/")
	parser.add_argument("--max-seq-length", default=128, type=int)
	parser.add_argument("--num-processes", default=5, type=int)
	parser.add_argument("--do-lower-case", dest="do_lower_case", action="store_true")
	parser.add_argument("--no-lower-case", dest="do_lower_case", action="store_false")
	parser.add_argument("--random", action="store_true")
	parser.add_argument("--epoch", default="1", type=str)
	parser.set_defaults(do_lower_case=True)
	args = parser.parse_args()

	utils.rmkdir(os.path.join("./dataset/", f"wordnet{args.epoch}_tfrecords"))
	owt_dir = os.path.join(args.data_dir, "openwebtext")

	print("Writing tf examples")
	fname = sorted(tf.io.gfile.listdir(owt_dir))

	if args.num_processes == 1:
		write_examples(0, args)
	else:
		jobs = []
		for job_id in range(args.num_processes):
			fnames = [f for (i, f) in enumerate(fname) if i%args.num_processes == job_id]
			random.shuffle(fnames)
			
			job = multiprocessing.Process(target=write_examples, args=(job_id, fnames, owt_dir, args))
			jobs.append(job)
			job.start()
		for job in jobs:
			job.join()

if __name__ == "__main__":
	main()
