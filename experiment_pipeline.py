import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from math import exp, sqrt, log
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import math


def load_data(file_path):

    data = pd.read_csv(file_path)
    return data

def see_labels(data):
    label_column = 'labels'
    set_labels = set(data[label_column])
    print(f"Unique labels in the dataset: {set_labels}")
    print("Column names in the dataset:")
    print(data.columns.tolist())

def names(data):
    names = []
    for i in range(len(data)):
        names.append(data.loc[i, "name"])
    names = list(set(names))
    return names

def phones(data):
    phones = []
    for i in range(len(data)):
        phones.append(data.loc[i, "phone"])
    phones = list(set(phones))
    return phones

def emails(data):
    emails = []
    for i in range(len(data)):
        emails.append(data.loc[i, "email"])
    emails = list(set(emails))
    return emails

def data_preparer(data, i, random_data):
    original_text = data.loc[i, "text"]
    name = data.loc[i, "name"]
    phone = data.loc[i, "phone"]
    email = data.loc[i, "email"]

    original_sentences = original_text.split('.')
    new_sentences = []
    for i in range(len(original_sentences)):
        if name in original_sentences[i]:
            new_sentences.append(original_sentences[i])
        elif phone in original_sentences[i]:
            new_sentences.append(original_sentences[i])
        elif email in original_sentences[i]:
            new_sentences.append(original_sentences[i])
        else:
            pass
    original_text = '.'.join(new_sentences)

    name_anonymized = original_text.replace(name, "[NAME]")
    phone_anonymized = name_anonymized.replace(phone, "[PHONE]")
    email_anonymized = phone_anonymized.replace(email, "[EMAIL]")

    random_name = random_data[0]
    random_phone = random_data[1]
    random_email = random_data[2]

    name_randomized = original_text.replace(name, random_name)
    phone_randomized = name_randomized.replace(phone, random_phone)
    email_randomized = phone_randomized.replace(email, random_email)


    return name_anonymized, phone_anonymized, email_anonymized, name_randomized, phone_randomized, email_randomized


data = load_data('new_data.csv')
names1 = names(data)
phones1 = phones(data)
emails1 = emails(data)






import random
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_drift_bounds(i, names1, phones1, emails1, epsilon=1.9):



    # Original sentence with two people mentioned multiple times
    original_text = data.loc[i, "text"]
    # randomly select name, phone, and email
    random_name = random.choice(names1)
    random_phone = random.choice(phones1)
    random_email = random.choice(emails1)
    random_data = [random_name, random_phone, random_email]
    prepared_data = data_preparer(data, i, random_data)
    name_anonymized, phone_anonymized, email_anonymized, name_randomized, phone_randomized, email_randomized = prepared_data



    # Anonymization: replace one full name at a time
    anon_steps = [
        original_text,
        name_anonymized,
        phone_anonymized]

    # Randomization: replace one full name at a time
    rand_steps = [
        original_text,
        name_randomized,
        phone_randomized]
    # Encode embeddings
    anon_embeddings = [model.encode(i) for i in anon_steps]
    rand_embeddings = [model.encode(i) for i in rand_steps]

    # Compute drift
    def compute_drift_bounds1(embeddings, label):
        c_squared = []
        for i in range(len(embeddings) - 1):
            c = norm(embeddings[i + 1] - embeddings[i])
            print(c)
            c_squared.append(c**2)
            print(f"{label} step {i}: Δ = {c:.4f}, Δ² = {c**2:.6f}")
        return c_squared, sum(c_squared)

    anon_c2, anon_sum = compute_drift_bounds1(anon_embeddings, "Anonymization")
    rand_c2, rand_sum = compute_drift_bounds1(rand_embeddings, "Randomization")

    # Azuma–Hoeffding bound
    def azuma_bound(epsilon, sum_c2):
        bound = 2 * math.exp(-epsilon**2 / (2 * sum_c2))
        return min(1.0, bound)

    epsilon = epsilon
    print(f"\nAzuma–Hoeffding upper bound (ε = {epsilon}):")
    print(f"Anonymization: P(|M_n - M_0| ≥ ε) ≤ {azuma_bound(epsilon, anon_sum):.6f}")
    print(f"Randomization:  P(|M_n - M_0| ≥ ε) ≤ {azuma_bound(epsilon, rand_sum):.6f}")
    return azuma_bound(epsilon, anon_sum), azuma_bound(epsilon, rand_sum)


def plot_averages(averages_anon, averages_rand, num, type):
    
    plt.figure(figsize=(10, 5))
    plt.plot(averages_anon, label='Anonymization')
    plt.plot(averages_rand, label='Randomization')
    plt.ylim(0, 1)
    plt.xlabel('Example Index')
    plt.ylabel('Average Azuma–Hoeffding Bound')
    plt.title(f'Average Azuma–Hoeffding Bounds (Experiment {num})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'azuma_hoeffding_experiments_{num}_{type}.png')
    plt.show()






# List of examples [(personA, personB), (randomA, randomB)]
# examples = [names1[i:i+4] for i in range(0, len(names1)-4, 4)]



final_anon_bounds = []
final_rand_bounds = []

epsilons = [1.9, 2.0, 2.1, 2.2, 2.3]
print("\n=== Experiment Across Multiple Sentences ===")
for k, epsilon in enumerate(epsilons):
    anon_bounds = []
    rand_bounds = []
    averages_anon  = []
    averages_rand = []
    for i in range(len(data)):
        # Using compute_drift_bounds function
        print(f"\nExample {i+1}/{len(data)}")
        print(f"\nExample {i+1}: {data.loc[i, 'name']} & {data.loc[i, 'phone']}")
        # --- Anonymization ---
        anon_bound, rand_bound = compute_drift_bounds(i, names1, phones1, emails1, epsilon=epsilon)
        anon_bounds.append(anon_bound)
        rand_bounds.append(rand_bound)
        print(f"  Anonymization total drift²: {anon_bound:.6f}")
        print(f"  Randomization total drift²:  {rand_bound:.6f}")
        
        averages_anon.append(sum(anon_bounds) / len(anon_bounds) if anon_bounds else 0)
        averages_rand.append(sum(rand_bounds) / len(rand_bounds) if rand_bounds else 0)
        # anon_bounds.append(anon_bound)
        # rand_bounds.append(rand_bound)




    # print(len(anon_bounds))
    # print(len(rand_bounds))
    avg_anon = sum(anon_bounds) / len(anon_bounds)
    avg_rand = sum(rand_bounds) / len(rand_bounds)
    final_anon_bounds.append(avg_anon)
    final_rand_bounds.append(avg_rand)
    print("\n=== Final Summary ===")
    print(f"Average Azuma–Hoeffding bound for anonymization (ε={epsilon}): {avg_anon:.6f}")
    print(f"Average Azuma–Hoeffding bound for randomization (ε={epsilon}): {avg_rand:.6f}")
    plot_averages(averages_anon, averages_rand, k+1, type='avg')

# Final results for table
print("\n=== Final Results for Table ===")
print("Epsilon\tAnonymization Bound\tRandomization Bound")
for i, epsilon in enumerate(epsilons):
    print(f"{epsilon:.1f}\t{final_anon_bounds[i]:.6f}\t\t{final_rand_bounds[i]:.6f}")

# Plot epsilon vs Azuma–Hoeffding bounds
plt.figure(figsize=(10, 5))
plt.plot(epsilons[::-1], final_anon_bounds, label='Anonymization', marker='o')
plt.plot(epsilons[::-1], final_rand_bounds, label='Randomization', marker='o')
plt.ylim(0, 1)
plt.xlabel('Epsilon (ε)')
plt.ylabel('Average Azuma–Hoeffding Bound')
plt.title('Epsilon vs Azuma–Hoeffding Bounds')
plt.legend()
plt.grid(True)
plt.savefig('epsilons_vs_azuma_hoeffding_bounds.png')
plt.show()

# final_bounds_df = pd.DataFrame({
#     'Epsilon': epsilons,
#     'Anonymization Bound': final_anon_bounds,
#     'Randomization Bound': final_rand_bounds
# })
# final_bounds_df.to_csv('final_azuma_hoeffding_bounds.csv', index=False)
# # Save the averages to a CSV file
# averages_df = pd.DataFrame({
#     'Example Index': list(range(1, 1001)),
#     'Anonymization Average': averages_anon,
#     'Randomization Average': averages_rand
# })
# averages_df.to_csv('averages_azuma_hoeffding_bounds.csv', index=False)
# # Save the anonymization and randomization bounds to a CSV file
# bounds_df = pd.DataFrame({
#     'Example Index': list(range(1, 1001)),
#     'Anonymization Bound': anon_bounds,
#     'Randomization Bound': rand_bounds
# })
# bounds_df.to_csv('bounds_azuma_hoeffding_bounds.csv', index=False)

