# %%
"""
Generate 2-number samples for curriculum learning on the countdown task.
Each sample is a tuple (a, b, t) where t is the result of one operation on a and b.
Operations: a+b, a-b, a*b, a/b
All values must be positive integers, t is between 10 and 100.
"""

import random

def generate_addition_samples(n=100, min_t=10, max_t=99):
    """a + b = t"""
    samples = []
    for _ in range(n):
        t = random.randint(min_t, max_t)
        a = random.randint(1, t - 1)
        b = t - a
        samples.append((a, b, t))
    return samples

def generate_subtraction_samples(n=100, min_t=10, max_t=99, max_val=99):
    """a - b = t"""
    samples = []
    while len(samples) < n:
        t = random.randint(min_t, max_t)
        # a = t + b, and a <= max_val, so b <= max_val - t
        max_b = max_val - t
        if max_b < 1:
            continue
        b = random.randint(1, max_b)
        a = t + b
        samples.append((a, b, t))
    return samples

def generate_multiplication_samples(n=100, min_t=10, max_t=99, max_val=99):
    """a * b = t, where a >= 2 and b >= 2"""
    samples = []
    while len(samples) < n:
        t = random.randint(min_t, max_t)
        # find divisors of t that are >= 2 and leave b >= 2
        divisors = [i for i in range(2, t + 1) if t % i == 0 and t // i >= 2]
        if not divisors:
            continue
        a = random.choice(divisors)
        b = t // a
        samples.append((a, b, t))
    return samples

def generate_division_samples(n=100, min_t=10, max_t=99, max_val=99):
    """a / b = t, where b >= 2"""
    samples = []
    while len(samples) < n:
        t = random.randint(min_t, max_t)
        # a = b * t, and a <= max_val, so b <= max_val // t
        max_b = max_val // t
        if max_b < 2:
            continue
        b = random.randint(2, max_b)
        a = b * t
        samples.append((a, b, t))
    return samples

# %%
n = 1000
addition_samples = generate_addition_samples(n)
subtraction_samples = generate_subtraction_samples(n)
multiplication_samples = generate_multiplication_samples(n)
division_samples = generate_division_samples(n)

# %%
from datasets import load_dataset, Dataset, DatasetDict

# %%
original_data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")["train"]

# %%
# Randomly sample 20k for train and 4k for test from original data
random.seed(42)
indices = list(range(len(original_data)))
random.shuffle(indices)

train_indices = indices[:20000]
test_indices = indices[20000:24000]

train_samples = [original_data[i] for i in train_indices]
test_samples = [original_data[i] for i in test_indices]

# %%
# Convert curriculum samples (a, b, t) to the same format as original data: {'target': t, 'nums': [a, b]}
def convert_to_dataset_format(samples):
    return [{'target': t, 'nums': [a, b]} for a, b, t in samples]

# Add difficulty field to each sample (difficulty = len(nums))
def add_difficulty(samples):
    for sample in samples:
        sample['difficulty'] = len(sample['nums'])
    return samples

addition_data = convert_to_dataset_format(addition_samples)
subtraction_data = convert_to_dataset_format(subtraction_samples)
multiplication_data = convert_to_dataset_format(multiplication_samples)
division_data = convert_to_dataset_format(division_samples)

# %%
# Add curriculum samples to training set
train_samples.extend(addition_data)
train_samples.extend(subtraction_data)
train_samples.extend(multiplication_data)
train_samples.extend(division_data)

# Shuffle the training set
random.shuffle(train_samples)

# %%
# Add difficulty field to both train and test samples
train_samples = add_difficulty(train_samples)
test_samples = add_difficulty(test_samples)

# %%
# Create HuggingFace datasets
train_dataset = Dataset.from_list(train_samples)
test_dataset = Dataset.from_list(test_samples)

print(f"Train dataset size: {len(train_dataset)}")  # 20000 + 4*1000 = 24000
print(f"Test dataset size: {len(test_dataset)}")    # 4000

# %%
train_dataset[0]
# %%

# %%
# Create unified DatasetDict with train and test splits
dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

print(dataset)

# %%
# Upload to HuggingFace Hub
dataset.push_to_hub("ozgur-celik/countdown_cl")
print("Dataset uploaded to https://huggingface.co/datasets/ozgur-celik/countdown_cl")

# %%