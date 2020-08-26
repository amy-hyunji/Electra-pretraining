def write_examples(job_id, args):
	if args.random:
		job_tmp_dir = os.path.join(args.data_dir, f"random{args.epoch}", "job_"+str(job_id))
	else:
		job_tmp_dir = os.path.join(args.data_dir, f"wordnet{args.epoch}", "job_"+str(job_id))
	
	print("#### dir: {} ####".format(job_tmp_dir))
	owt_dir = os.path.join(args.data_dir, "openwebtext")
	
	def log(*args):
		msg = " ".join(map(str, args))
		print("Job {}:".format(job_id), msg)

	log("Creating example writer")
	example_writer = build_pretraining_dataset.ExampleWriter(
			job_id = job_id,
			vocab_file = os.path.join(args.data_dir, "vocab.txt"),
			output_dir = os.path.join(args.data_dir, f"random{args.random}_tfrecords") if args.random else os.path.join(args.data_dir, f"wordnet{args.epoch}_tfrecords"),
			max_seq_length = args.max_seq_length,
			num_jobs = args.num_processes,
			blanks_separate_docs = False,
			do_lower_case = args.do_lower_case,
			random = args.random
			)
	log("Writing tf examples")
	fnames = sorted(tf.io.gfile.listdir(owt_dir))
	fnames = [f for (i, f) in enumerate(fnames) if i%args.num_processes == job_id]
	random.shuffle(fnames)
	
	if args.random:
		f = open("./dataset/word.txt", "r", encoding="utf-8", errors="ignore")
		total_tokens = f.readlines()
		f.close()	
	else:
		total_tokens = []

	start_time = time.time()
	log("Start time: {}".format(start_time))
	for file_no, fname in enumerate(fnames):
		if (file_no > 0 and file_no % 10 == 0):
			elapsed = time.time() - start_time
			log("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s, ETA: {:}s, {:} examples written".format(file_no, len(fnames), 100.0*file_no/len(fnames), int(elapsed), int((len(fnames) - file_no) / (file_no / elapsed)), example_writer.n_written))
		else:
			print("{}/{}".format(file_no, len(fnames)))
		utils.rmkdir(job_tmp_dir)
		with tarfile.open(os.path.join(owt_dir, fname)) as f:
			f.extractall(job_tmp_dir)
		extracted_files = tf.io.gfile.listdir(job_tmp_dir)
		random.shuffle(extracted_files)
		for i, txt_fnames in enumerate(extracted_files):
#print("iter over {}/{}".format(i, len(extracted_files)))
			example_writer.write_examples(os.path.join(job_tmp_dir, txt_fnames), total_tokens)
	example_writer.finish()
	log("Done!")

def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-dir", default="./dataset")
	parser.add_argument("--max-seq-length", default=128, type=int)
	parser.add_argument("--num-processes", default=8, type=int)
	parser.add_argument("--do-lower-case", dest="do_lower_case", action="store_true")
	parser.add_argument("--no-lower-case", dest="do_lower_case", action="store_false")
	parser.add_argument("--random", action="store_true")
	parser.add_argument("--epoch", default="1", type=str)
	parser.set_defaults(do_lower_case=True)
	args = parser.parse_args()

	if args.random:
		utils.rmkdir(os.path.join(args.data_dir, "random_tfrecords"))
	else:
		utils.rmkdir(os.path.join(args.data_dir, f"wordnet{args.epoch}_tfrecords"))

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

if __name__ == "__main__":
	import os
	os.system("cp ./tree.py /opt/conda/lib/python3.7/site-packages/pattern3/text/tree.py")
	print("##### Done copying tree.py ######")
	os.system("cat /opt/conda/lib/python3.7/site-packages/pattern3/text/tree.py")
	
	import argparse
	import multiprocessing
	import random
	import tarfile
	import time
	import tensorflow.compat.v1 as tf
	import sys

	import build_pretraining_dataset
	from util import utils, gen

	main()

