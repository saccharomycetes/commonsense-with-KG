# this code is used to generate the random x% of a data

import random
import jsonlines
import argparse
import random


def load_file(file_path):
    all_data = []
    with open(file_path) as f:
        for item in jsonlines.Reader(f):
            all_data.append(item)
    return all_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", default=None, type=int, required=True,help="the ratio of choose data")
    parser.add_argument("--data_i", default=None, type=str, required=True,help="where the data from")
    parser.add_argument("--data_o", default=None, type=str, required=True,help="where the data write to")
    args = parser.parse_args() 
    choose_ratio = args.ratio
    datas = load_file(args.data_i)
    datas = random.sample(datas, int(len(datas)*args.ratio/100))
    with jsonlines.open(args.data_o, "w")as f:
        for data in datas:
            f.write(data)
if __name__ == "__main__":
    main()