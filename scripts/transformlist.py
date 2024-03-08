import pandas as pd
import argparse
import json

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='file')
	parser.add_argument('--file', required=True,help='path to Test file')
	args = parser.parse_args()
	data = pd.read_csv(args.file, names=["datasets"])
	print(data.head())
	dicr = {"datasets":list(data["datasets"].unique())}
	print(dicr)
	with open(f"{args.file}.json","w") as h:
		json.dump(dicr,h, indent=4)