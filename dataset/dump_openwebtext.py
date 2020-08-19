import tarfile
import os
from tqdm import tqdm, notebook

path = "./openwebtext/"
txtpath = "./owt_txt/"
savepath = "./n_owt.txt"

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
		
