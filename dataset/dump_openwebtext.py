import tarfile
import os
from tqdm import tqdm, notebook

path = "/Users/user/Desktop/lee/electra/dataset/openwebtext"
txtpath = "./owt_txt/"
savepath = "./n_owt.txt"

def extract_one():
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
			
def extract_all():
	xzFiles = os.listdir(path)
	if not os.path.exists(txtpath):
		os.mkdir(txtpath)
	for i in tqdm(range(len(xzFiles))):
		xzfile = xzFiles[i]
		with tarfile.open(os.path.join(path, xzfile)) as f:
			f.extractall(txtpath)


if __name__ == "__main__":
	extract_all()
