import os
with open('train.py', 'w', encoding='utf-8') as outfile:
    for i in range(1, 6):
        with open(f'chunk{i}.py', 'r', encoding='utf-8') as infile:
            outfile.write(infile.read())
print("Combined chunks into train.py successfully.")
