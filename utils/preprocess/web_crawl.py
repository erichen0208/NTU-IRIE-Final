import json
import re
import requests
from bs4 import BeautifulSoup

with open('../../chlaw.json/ChLaw.json', 'r', encoding='utf-8-sig') as file:
    data = json.load(file)

laws_info = {}

for law in data["Laws"]:
    law_name = law["LawName"]
    law_url = law["LawURL"]
    
    pcode_match = re.search(r'pcode=([A-Z0-9]+)', law_url)
    pcode = pcode_match.group(1) if pcode_match else None
    
    laws_info[law_name] = pcode

# 定义函数，从网站抓取法律条文的解释文字
def fetch_law_example(pcode, flno):
    url = f"https://law.moj.gov.tw/LawClass/LawSingleRela.aspx?PCODE={pcode}&FLNO={flno}&ty=J"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.select('div.text-con div.row div.text-pre')
    example = []
    for row in rows:
        clean_text = row.text.replace("\r\n", "").strip()
        example.append(clean_text)
    return example

input_file = "law.jsonl"
output_file = "law_with_examples.jsonl"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line) 
        provision = data.get("provision", "")
        
        if "第" in provision and "條" in provision:
            law_name_match = re.match(r"([^\d]+)第(\d+)條", provision) 
            if law_name_match:
                law_name = law_name_match.group(1).strip()  
                flno = law_name_match.group(2).strip() 

                pcode = laws_info.get(law_name)
                if pcode: 
                    example = fetch_law_example(pcode, flno)
                    data["example"] = example  
                else:
                    data["example"] = []  
        else:
            data["example"] = []  

        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

print("Complete!", output_file)
