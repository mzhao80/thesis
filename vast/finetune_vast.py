import os
import pandas as pd
import torch
import logging
from typing import List
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
TRAIN_CSV_PATH = "vast/zero-shot-stance/data/VAST/vast_train.csv"  # Adjust to your path
VAL_CSV_PATH = "vast/zero-shot-stance/data/VAST/vast_test.csv"     # Adjust to your path
MODEL_NAME = "nvidia/NV-Embed-v2"  # Pretrained HF model on Hugging Face
OUTPUT_DIR = "./vast_finetuned_model_hf"  # Where to save the fine-tuned model

EPOCHS = 10
BATCH_SIZE = 16             # Adjust per your GPU memory
LEARNING_RATE = 1e-5
GRADIENT_ACCUM_STEPS = 2
WARMUP_RATIO = 0.1


logging.basicConfig(level=logging.INFO)




# -----------------------------------------------------------
# Step 1: Load CSV data
# -----------------------------------------------------------
def read_vast_data(csv_file: str):
    """
    Reads a VAST csv file and returns a list of dicts:
    [
      {
        'text': <full comment text from 'post'>,
        'topic': <new_topic or 'no_controversial_topic'>,
      },
      ...
    ]
    We also return a set of all distinct topics, including "no_controversial_topic".
    """
    df = pd.read_csv(csv_file)
    # We assume the CSV has columns: "post", "new_topic", "label"
    # label = 0 (con), 1 (pro), 2 (neutral).
    texts_topics = []
    topic_set = set()
    
    for row in df.itertuples(index=False):
        text = getattr(row, "post")
        label = getattr(row, "label")
        raw_topic = getattr(row, "new_topic")

        # For stance label=2 => treat as no controversial topic in this post
        if label == 2:
            # No controversial topic
            final_topic = "no_controversial_topic"
        else:
            # Use the new_topic as the label
            final_topic = str(raw_topic).strip().lower()
            if not final_topic:
                final_topic = "no_controversial_topic"

        texts_topics.append({"text": text, "topic": final_topic})
        topic_set.add(final_topic)

    return texts_topics, topic_set

# -----------------------------------------------------------
# Step 2: Construct Label Mappings & InputExamples
# -----------------------------------------------------------
def create_input_examples(
    items: List[dict],
    topic2label: dict,
):
    """
    Convert the loaded items into a list of sentence_transformers.InputExample,
    suitable for softmax classification. Each example has:
       - texts=[ the original text ]
       - label = integer label for the topic
    """
    examples = []
    for it in items:
        text = it["text"]
        topic = it["topic"]
        label_id = topic2label[topic]
        examples.append(InputExample(texts=[text], label=label_id))
    return examples


# -----------------------------------------------------------
# Main Fine-Tuning Routine
# -----------------------------------------------------------
def main():
    # Load training data
    train_data, train_topics = read_vast_data(TRAIN_CSV_PATH)
    # Load test data (optional, for evaluation)
    test_data, test_topics = read_vast_data(TEST_CSV_PATH)
    
    # Build a global set of all possible topics from train+test
    # (Assuming you want the model to handle test topics that appear in train. 
    #  If new topics appear only in test, that complicates classification. 
    #  For demonstration, we'll unify them.)
    all_topics = set(train_topics) | set(test_topics)
    all_topics = sorted(list(all_topics))
    
    # Map each topic string to an integer label
    topic2label = {t: i for i, t in enumerate(all_topics)}
    num_labels = len(all_topics)
    
    print(f"Number of distinct topics (including 'no_controversial_topic'): {num_labels}")

    # Create InputExamples
    train_examples = create_input_examples(train_data, topic2label)
    test_examples = create_input_examples(test_data, topic2label)

    # Dataloaders
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=EVAL_BATCH_SIZE)

    # Load the base model with trust_remote_code=True
    # This ensures that custom code from the 'nvidia/NV-Embed-v2' repo is trusted.
    # 2) Create SentenceTransformer with model_kwargs
    #    This is where you can set torch_dtype, attention implementations, etc.
    print("Initializing SentenceTransformer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Construct model_kwargs
    chosen_dtype = 'torch.bfloat16'
    
    model_kwargs = {
        "torch_dtype": chosen_dtype,
    }
    
    # Note: trust_remote_code=True is needed if the model is custom-coded
    model = SentenceTransformer(
        model_name, 
        trust_remote_code=True,
        model_kwargs=model_kwargs
    ).to(device)
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    # Prepare a softmax classification head
    # The dimension is taken from the base model's sentence embeddings
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=num_labels
    )

    # -----------------------------------------------------------
    # Step 3: Fine-tune the Model
    # -----------------------------------------------------------
    # You can pass an eval_data if you want validation metrics every epoch.
    # For example:
    # model.fit(
    #     train_objectives=[(train_dataloader, train_loss)],
    #     evaluator=LabelAccuracyEvaluator(test_dataloader),
    #     epochs=EPOCHS,
    #     warmup_steps=int(len(train_dataloader) * EPOCHS * WARMUP_RATIO),
    #     output_path=OUTPUT_DIR
    # )
    #
    # But here is a simpler version (no evaluation callback in the middle):
    warmup_steps = int(len(train_dataloader) * EPOCHS * WARMUP_RATIO)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        optimizer_params={"lr": LEARNING_RATE},
        warmup_steps=warmup_steps,
        output_path=OUTPUT_DIR
    )

    # After training finishes, you can evaluate on test set manually
    # (Optional) Quick check: measure classification accuracy
    evaluate_accuracy(model, test_dataloader)

    # The fine-tuned model is saved in OUTPUT_DIR
    # You can later load it via: SentenceTransformer(OUTPUT_DIR)


def evaluate_accuracy(model: SentenceTransformer, dataloader: DataLoader):
    """
    Simple function that computes accuracy by comparing
    the argmax of the model's predicted logits vs. ground truth label.
    """
    correct, total = 0, 0
    # We use the same SoftmaxLoss or replicate the logic
    # to get logits from the final classification layer:
    with torch.no_grad():
        for batch in dataloader:
            # batch is a list of InputExample objects turned into a dictionary
            # by the DataLoader + SentenceTransformers collate_fn
            input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
            # Move to GPU if available
            input_ids = input_ids.to(model._target_device)
            attention_mask = attention_mask.to(model._target_device)
            labels = labels.to(model._target_device)

            # forward() from SentenceTransformer returns a dictionary containing 'sentence_embedding'
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            # The SoftmaxLoss layer is typically appended in training,
            # so to replicate the classification logits, you can do:
            emb = output["sentence_embedding"]
            # [batch_size, hidden_dim] -> [batch_size, num_labels]
            # We'll mimic the internal classification layer
            # by retrieving the SoftmaxLoss if in the model modules:
            # For clarity, just do a quick linear:
            #   classification_head = model[<some module index>]
            # But we have to be sure we replicate the same head used in training.

            # If you do not want to rummage through the model layers, you can
            # simply re-initialize the same SoftmaxLoss in inference mode:
            #   train_loss = losses.SoftmaxLoss(...)
            # then call train_loss.forward(emb, labels=None, return_logits=True).
            # For demonstration, let's do the "loose" approach:

            # Example: find the softmax layer from the modules (this is a bit hacky).
            # A more robust approach: keep your SoftmaxLoss as a separate module you can query.
            # This code attempts to locate it by type:
            classification_layer = None
            for m in model.modules():
                if isinstance(m, losses.SoftmaxLoss):
                    classification_layer = m
                    break

            if classification_layer is None:
                raise ValueError("Could not find the SoftmaxLoss layer in the model modules. "
                                 "Ensure you're using the same environment as training.")
            
            logits = classification_layer.linear(emb)  # shape [batch_size, num_labels]
            predictions = torch.argmax(logits, dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    print(f"Test Accuracy: {correct / total:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
