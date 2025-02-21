import pickle

with open('zero-shot-stance/data/VAST/wiki_dict.pkl', 'rb') as file:
    data = pickle.load(file)
    i = 0
    for key, value in data.items():
        print(f"{key}: {value}")
        i += 1
        if i == 5:
            break
