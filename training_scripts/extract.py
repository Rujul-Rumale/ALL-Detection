import os
log_path = r'C:\Users\rujul\.gemini\antigravity\brain\58498964-554c-4f7d-a156-78b98b342518\.system_generated\logs\overview.txt'
with open(log_path, 'r', encoding='utf-8') as f: content = f.read()

start_marker = '"""\ntrain.py\n========\nPhase 1'
end_marker = 'if __name__ == "__main__":\n    main()\n'

start_idx = content.rfind(start_marker)
end_idx = content.rfind(end_marker)

print(start_idx, end_idx)
if start_idx != -1 and end_idx != -1:
    code = content[start_idx:end_idx + len(end_marker)]
    with open(r'c:\Open Source\leukiemea\training_scripts\train.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("Wrote successfully")
else:
    print("Marks not found")
