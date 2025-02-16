#!/usr/bin/env python
import argparse
import csv
import time
from openai import OpenAI
import api_keys
from sklearn.metrics import f1_score

def classify_stance(document, topic, client, max_attempts=5):
    """
    Constructs a prompt for GPT-4o-mini to classify the stance of a document on a topic.
    It instructs the model to answer with exactly one word: "pro", "con", or "neutral".
    Retries up to max_attempts until a valid response is received.
    """
    prompt = (
        "Given the following document and topic, determine the stance of the document on the topic.\n"
        f"Document: {document}\n"
        f"Topic: {topic}\n"
        "Answer with one word: 'pro', 'con', or 'neutral'."
    )
    messages = [
        {"role": "system",
         "content": (
             "You are a helpful assistant that classifies the stance of a congressional speech with respect to a policy topic. "
             "Respond with exactly one word: 'pro', 'con', or 'neutral'."
         )},
        {"role": "user", "content": prompt}
    ]
    
    attempt = 0
    while attempt < max_attempts:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            content = response.choices[0].message.content.strip().lower()
            # Look for one of the valid labels in the response.
            for label, key in zip(["pro", "con", "neutral"], [1, 0, 2]):
                if label == content:
                    return key
            raise ValueError(f"Error on attempt {attempt+1}: {content}")
        except Exception as e:
            print(e)
            attempt += 1
    
    return ""

def load_validation_set(csv_file):
    """
    Loads the validation CSV file. The CSV must contain the following columns:
      - document
      - topic
      - label   (expected to be one of "pro", "con", or "neutral")
      - seen    (0 for zero-shot, 1 for few-shot)
    Returns a list of dictionaries.
    """
    data = []
    with open(csv_file, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "document": row["post"],
                "topic": row["topic_str"],
                "label": int(row["label"]),
                "seen": int(row["seen?"])
            })
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Use GPT-4o-mini to classify a validation set and report F1 metrics."
    )
    parser.add_argument("--val_csv", type=str, default="zero-shot-stance/data/VAST/vast_dev.csv",
                        help="Path to the validation CSV file.")
    args = parser.parse_args()

    print("Loading validation data...")
    data = load_validation_set(args.val_csv)

    # Prepare lists to collect true and predicted labels.
    all_true = []
    all_pred = []

    zero_true, zero_pred = [], []
    few_true, few_pred = [], []

    # For binary evaluation on "pro" and "con"
    overall_pro_true, overall_pro_pred = [], []
    overall_con_true, overall_con_pred = [], []
    zero_pro_true, zero_pro_pred = [], []
    zero_con_true, zero_con_pred = [], []
    few_pro_true, few_pro_pred = [], []
    few_con_true, few_con_pred = [], []

    # Initialize OpenAI client
    client = OpenAI(api_key=api_keys.OPENAI_API_KEY)
    key = {0: "con", 1: "pro", 2: "neutral"}

    print("Classifying validation samples with GPT-4o-mini...")
    for idx, sample in enumerate(data):
        document = sample["document"]
        topic = sample["topic"]
        true_label = sample["label"]
        seen = sample["seen"]

        pred_label = classify_stance(document, topic, client)
        all_true.append(true_label)
        all_pred.append(pred_label)

        # Group by zero-shot and few-shot
        if seen == 0:
            zero_true.append(true_label)
            zero_pred.append(pred_label)
        else:
            few_true.append(true_label)
            few_pred.append(pred_label)

        # For stance-specific (only consider samples where the true label is pro or con)
        if true_label == 1:
            overall_pro_true.append(true_label)
            overall_pro_pred.append(pred_label)
            if seen == 0:
                zero_pro_true.append(true_label)
                zero_pro_pred.append(pred_label)
            else:
                few_pro_true.append(true_label)
                few_pro_pred.append(pred_label)
        elif true_label == 0:
            overall_con_true.append(true_label)
            overall_con_pred.append(pred_label)
            if seen == 0:
                zero_con_true.append(true_label)
                zero_con_pred.append(pred_label)
            else:
                few_con_true.append(true_label)
                few_con_pred.append(pred_label)
        
        # Print progress (showing beginning of document/topic)
        print(f"[{idx+1}/{len(data)}] True: {key[true_label]}  Pred: {key[pred_label]}  Seen: {seen}")

    # Calculate overall macro F1
    overall_f1 = f1_score(all_true, all_pred, average="macro")
    print("\n=== Overall Evaluation ===")
    print(f"Overall Macro F1: {overall_f1:.4f}")

    # Calculate zero-shot and few-shot macro F1 scores.
    if zero_true:
        zero_f1 = f1_score(zero_true, zero_pred, average="macro")
        print(f"Zero-shot Macro F1: {zero_f1:.4f}")
    else:
        print("No zero-shot samples available.")
    if few_true:
        few_f1 = f1_score(few_true, few_pred, average="macro")
        print(f"Few-shot Macro F1: {few_f1:.4f}")
    else:
        print("No few-shot samples available.")

    # For binary F1 on pro and con:
    def safe_binary_f1(true_list, pred_list, pos_label):
        return f1_score(true_list, pred_list, labels=[pos_label], average=None) if true_list else None

    overall_f1_pro = safe_binary_f1(overall_pro_true, overall_pro_pred, pos_label=1)
    overall_f1_con = safe_binary_f1(overall_con_true, overall_con_pred, pos_label=0)
    zero_f1_pro = safe_binary_f1(zero_pro_true, zero_pro_pred, pos_label=1)
    zero_f1_con = safe_binary_f1(zero_con_true, zero_con_pred, pos_label=0)
    few_f1_pro = safe_binary_f1(few_pro_true, few_pro_pred, pos_label=1)
    few_f1_con = safe_binary_f1(few_con_true, few_con_pred, pos_label=0)

    print("\n=== By Stance (Binary F1 Scores) ===")
    if overall_f1_pro is not None:
        print(f"Overall Pro F1: {overall_f1_pro[0]:.4f}")
    else:
        print("No overall 'pro' samples.")
    if overall_f1_con is not None:
        print(f"Overall Con F1: {overall_f1_con[0]:.4f}")
    else:
        print("No overall 'con' samples.")

    if zero_f1_pro is not None:
        print(f"Zero-shot Pro F1: {zero_f1_pro[0]:.4f}")
    else:
        print("No zero-shot 'pro' samples.")
    if zero_f1_con is not None:
        print(f"Zero-shot Con F1: {zero_f1_con[0]:.4f}")
    else:
        print("No zero-shot 'con' samples.")

    if few_f1_pro is not None:
        print(f"Few-shot Pro F1: {few_f1_pro[0]:.4f}")
    else:
        print("No few-shot 'pro' samples.")
    if few_f1_con is not None:
        print(f"Few-shot Con F1: {few_f1_con[0]:.4f}")
    else:
        print("No few-shot 'con' samples.")

if __name__ == "__main__":
    main()