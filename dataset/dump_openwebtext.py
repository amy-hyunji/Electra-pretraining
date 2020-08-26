import tarfile
import os
from tqdm import tqdm, notebook

def dump_all():
	path = "./openwebtext/"
	txtpath = "./owt_txt/"
	savepath = "./owt.txt"

	xzFiles = os.listdir(path)
	with open(savepath, "w") as owt: 
		for i in tqdm(range(len(xzFiles))):
			xzfile = xzFiles[i]
			with tarfile.open(os.path.join(path, xzfile)) as f:
				if not os.path.exists(txtpath):
					os.mkdir(txtpath)
				f.extractall(txtpath)
			txtList = os.listdir(txtpath)
			for txt in txtList:
				temp = open(os.path.join(txtpath, txt))
				try:
					owt.write(temp.read())
				except:
					lines = temp.readlines()
					for line in lines:
						owt.write(line)
			os.system(f"rm -rf {txtpath}")
	

def dump_sep():
	path = "./openwebtext/"
	txtpath = "./owt_txt/"

	if not os.path.exists(txtpath):
		os.mkdir(txtpath)

	xzFiles = os.listdir(path)
	for i in tqdm(range(len(xzFiles))):
		xzfile = xzFiles[i]
		with tarfile.open(os.path.join(path, xzfile)) as f:
			f.extractall(txtpath)


if __name__=="__main__":
	dump_sep()
