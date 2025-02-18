
import argparse
from utils.train import model
from utils.evaluate import translate_sentence

parser = argparse.ArgumentParser(description="Bangla-English Translation System")
parser.add_argument("--mode", type=str, required=True, help="train or translate")
parser.add_argument("--sentence", type=str, help="Sentence to translate")

args = parser.parse_args()

if args.mode == "train":
    model.fit()
elif args.mode == "translate" and args.sentence:
    print(translate_sentence(args.sentence))