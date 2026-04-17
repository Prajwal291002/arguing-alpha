import os
file_path = "data/raw_filings/A/2016/0001090872-16-000076_10-Q.html"
with open(file_path, "r", encoding="utf-8") as file:
    for i in range(100):
        print(file.readline())