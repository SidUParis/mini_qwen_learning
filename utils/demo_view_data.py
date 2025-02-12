from datasets import load_dataset
import argparse 
parser = argparse.ArgumentParser()

parser.add_argument("type",help="please type which kinds of data you want to see",type=str,choices=['dpo','pt','sft'])
args = parser.parse_args()

if args.type == "dpo":
    dataset = load_dataset("parquet", data_files="mini_data/dpo/train-00000-of-00001.parquet", split="train")
    print(dataset[5])
elif args.type == 'sft':
    dataset = load_dataset("parquet", data_files="mini_data/sft/7M/train-00000-of-00075.parquet", split="train")
    print(dataset[9])
elif args.type == 'pt':
    dataset = load_dataset("parquet", data_files="mini_data/pt/accommodation_catering_hotel/chinese/high/rank_00000.parquet", split="train")
    print(dataset[0])









