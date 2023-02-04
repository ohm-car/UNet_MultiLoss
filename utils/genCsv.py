import csv
import numpy as np
from PIL import Image
import os
from pathlib import Path

def gen_perc(img):

	print(img.size)

	tp1 = 1*(img == 3)
	# with np.printoptions(threshold = np.inf):
	# 	print(img)

	tp2 = 1*(img == 3) + 1*(img == 1)
	# with np.printoptions(threshold = np.inf):
	# 	print(tp2)

	p1 = np.sum(tp1)/img.size
	p2 = np.sum(tp2)/img.size
	print('p1:', p1)
	print('p2:', p2)

	return p1, p2

def main():

	#get list of files from annotations folder

	cpath = Path(__file__)
	rpath = cpath.parent.parent.parent
	fpath = os.path.join(rpath, 'annotations/trimaps')
	# files = os.listdir(fpath)

	files = sorted(
    [
        os.path.join(fpath, fname)
        for fname in os.listdir(fpath)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

	#open csv

	filename = os.path.join(rpath, 'percs.csv')
	with open(filename, 'w') as csvfile:

		csvwriter = csv.writer(csvfile)

		for imgname in files:
			print(os.path.join(fpath, imgname))
			img = Image.open(os.path.join(fpath, imgname))

			p1, p2 = gen_perc(np.asarray(img))

			csvwriter.writerow([imgname.split('Thesis_Work')[1], str(p1), str(p2)])

		# csvfile.close()

	#for each file:
		# open image
		# note filename
		# calc percentage(s)
		# write to csv

if __name__ == '__main__':
	main()