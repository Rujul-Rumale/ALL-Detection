import re
import sys

print("LATENCY CLAIMS:")
print("---------------")
with open(r"c:\Open Source\leukiemea\paper\latex\alternate_tj_latex_template_ap\main.tex", "r", encoding='utf-8') as f:
    text = f.read()

for match in re.finditer(r'(.{0,50}latency.{0,50})', text, re.I):
    print(f"Match: {match.group(1).strip()}")

for match in re.finditer(r'(.{0,50}ms.{0,50})', text, re.I):
    print(f"Match: {match.group(1).strip()}")

print("\nVALIDATION CLAIMS:")
print("------------------")
for match in re.finditer(r'(.{0,50}fold.{0,50})', text, re.I):
    print(f"Match: {match.group(1).strip()}")

print("\nENSEMBLE CLAIMS:")
print("----------------")
for match in re.finditer(r'(.{0,50}ensemble.{0,50})', text, re.I):
    print(f"Match: {match.group(1).strip()}")
for match in re.finditer(r'(.{0,50}Ens.{0,50})', text, re.I):
    print(f"Match: {match.group(1).strip()}")

print("\nTHRESHOLD CLAIMS:")
print("-----------------")
for match in re.finditer(r'(.{0,50}threshold.{0,50})', text, re.I):
    print(f"Match: {match.group(1).strip()}")

for match in re.finditer(r'(.{0,50}Youden.{0,50})', text, re.I):
    print(f"Match: {match.group(1).strip()}")
