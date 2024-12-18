import json
import re


'''
[
    {
        "genere": "第一编",
        "provisions": [
            {"name": "XX法第一条", "content": "第一条的内容", "example": "第一条的例子"},
            {},
            {}
        ]  
    },
    {
        "genere": "第二编",
        "provisions": [
            {},
            {},
            {}
        ]
    }
]
'''

def replace_symbol(sentence):
    sentence = re.sub(r'\b\d+\s+', '', sentence)
    pattern = r'[^，。：；]*處新臺幣[^，。：；]*'
    sentence = re.sub(pattern, '。', sentence)
    pattern = r'[，。：；]+'
    sentence = re.sub(pattern, '。', sentence)
    return sentence

def main():
    provision_list = []
    current_provision = []
    genere = None
    delete_ct = 0
    ct = 0

    with open("../data/law_with_examples.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            ct += 1
            item = json.loads(line)
            provision_name = item['provision']
            provision_content = item['content']
            provision_example = item['example']

            # current_genere = provision_name.split('第')[0]
            # if current_genere != genere:
            #     if genere != None:
            #         print(len(current_provision))
            #         provision_list.append({"genere": genere, "provisions": current_provision})
            #         current_provision = []

            #     genere = current_genere
            
            if len(provision_example) == 0 and len(provision_content) < 25:
                delete_ct += 1
                continue
            provision_list.append(item)
            # provision_name = re.sub(r'\s+', '', provision_name)
            # provision_content = replace_symbol(provision_content)
            # provision_example = [re.sub(r'\s+', '', example) for example in provision_example]
            # current_provision.append({"name": provision_name, "content": provision_content, "example": provision_example}) 

    print(f"Total: {ct}, Delete: {delete_ct}")
    with open("./provision_list.jsonl", 'w', encoding='utf-8') as jsonl_file:
        for provision in provision_list:
            jsonl_file.write(json.dumps(provision, ensure_ascii=False) + '\n')
    # with open("./provision_list.py", 'w', encoding='utf-8') as py_file:
    #     py_file.write("provision_list = ")
    #     py_file.write(json.dumps(provision_list, ensure_ascii=False, indent=4))

def prepare_data_pretrain():
    data = []
    for i in range(len(provision_list)):
        data.append(provision_list[i]['genere'])
        for item in provision_list[i]['provisions']:
            data.append(item['content'])
            if (len(item['example']) > 0):
                data.extend(item['example'])
    
    with open("./data/train_data.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if(item['title']):
                data.append(item['title'])
            if(item['question']):   
                data.append(item['question'])

    with open("./data/test_data.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if(item['title']):
                data.append(item['title'])
            if(item['question']):   
                data.append(item['question'])
    
    with open("./data/pretrain_data.txt", 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line)

def provision_list_json():
    from data.py.provision_list import provision_list
    d = {}
    for i in range(len(provision_list)):
        for item in provision_list[i]["provisions"]:
            d[item["name"]] = {
                "content": item["content"],
                "example": item["example"]
            }

    with open("./data/provision.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(d, ensure_ascii=False, indent=4))
        
if __name__ == '__main__':
    # prepare_data_pretrain()
    # data = []
    # with open("./data/pretrain_data.txt", 'r', encoding='utf-8') as f:
    #     for line in f:
    #         if line == '\n':
    #             continue
    #         data.append(line)

    # with open("./data/pretrain_data.txt", 'w', encoding='utf-8') as f:
    #     for line in data:
    #         f.write(line)

    provision_list_json()