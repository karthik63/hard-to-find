import re
import json
import argparse

IO_match = re.compile(r'(?P<start>\d+)I-(?P<label>\S+)\s(?:(?P<end>\d+)I-(?P=label)\s)*')

def find_offsets(seqstr, match):
    annotations = set()
    for annotation in match.finditer(seqstr):
        start = int(annotation.group('start'))
        label = annotation.group('label')
        end = annotation.group('end')
        end = start + 1 if end is None else int(end) + 1
        annotations.add((start, end, label))
    return annotations

def compare_two_sequences(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError(f"{seq1}, {seq2} length not match")
    seqstr1 = " ".join([f"{i}I-{t}" if t != 'O' else t for i, t in enumerate(seq1)]) + " "
    seqstr2 = " ".join([f"{i}I-{t}" if t != 'O' else t for i, t in enumerate(seq2)]) + " "
    spans1 = find_offsets(seqstr1, IO_match)
    spans2 = find_offsets(seqstr2, IO_match)
    matched = spans1.intersection(spans2)
    return len(matched), len(spans1), len(spans2)

def process_txt_file(txt_file):
    labels = []
    with open(txt_file, "rt") as fp:
        label = []
        for line in fp:
            line = line.strip()
            if len(line) == 0:
                labels.append(label)
                label = []
            else:
                label.append(line.strip().split()[-1])
        if len(label) > 0:
            labels.append(label)
    return labels

def process_jsonl_file(jsonl_file):
    with open(jsonl_file, "rt") as fp:
        data = [json.loads(t) for t in fp]
    return [t['query']['label'] for t in data]

def compute_list_f1(annotations, predictions):
    matchings = [compare_two_sequences(a, p) for a,p in zip(annotations, predictions)]
    nmatch = sum([t[0] for t in matchings])
    ngold = sum([t[1] for t in matchings])
    npred = sum([t[2] for t in matchings])
    return {
        "precision": nmatch / npred,
        'ngold': ngold,
        'npred': npred,
        "recall": nmatch / ngold,
        "f1": nmatch * 2 / (npred + ngold)
    }
    

def supervised_f1(annotation_txt, prediction_txt):
    annotations = process_txt_file(annotation_txt)
    predictions = process_txt_file(prediction_txt)
    return compute_list_f1(annotations, predictions)
    
    

def fewshot_f1(annotation_jsonl, prediction_jsonl):
    annotations = process_jsonl_file(annotation_jsonl)
    predictions = process_jsonl_file(prediction_jsonl)
    results = [compute_list_f1(annotation, prediction) for annotation, prediction in zip(annotations, predictions)]
    return {
        key: sum([r[key] for r in results]) / len(results) for key in results[0]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-type", type=str, choices=["txt","jsonl"], help="txt for the supervised setting, jsonl for intra and inter settings")
    parser.add_argument("--annotation-file", type=str, help="path to the annotation file (from the dataset)")
    parser.add_argument("--prediction-file", type=str, help="path to your system output, make sure your output order is the same as the annotation file")
    args = parser.parse_args()
    if args.file_type == "txt":
        print(supervised_f1(args.annotation_file, args.prediction_file))
    else:
        print(fewshot_f1(args.annotation_file, args.prediction_file))


if __name__ == "__main__":
    main()