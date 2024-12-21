import torch
from transformers import pipeline
from data.py.train_list import train_list
import json

# Few-shot examples
few_shot_examples = [
    {
        "role": "user",
        "content": "公司辦展覽欲雇用工讀生看顧器材，分為兩個時段各四個小時\n\r\nQ1, 若工讀生自願連續工作八個小時不休息，這樣是否違反勞基法？ 那若雙方協議可以嗎？ 如果不行，休息時間是否需要照算薪水？（例如來上班八個小時，安排休息一個小時，薪水是算八小時還是七小時）\n\r\nQ2, 工讀生若在國定假日上班，薪水也是否給雙倍？\n\r\nQ3, 如果要跟工讀生訂定工作契約來確保工讀生須對設備保管責任，哪類相關契約可以參考？",
    },
    {
        "role": "assistant",
        "content": "勞基法、協議、休息、薪水。"
    },
    {
        "role": "user",
        "content": "最近看新聞，看到有消費者PO文爆料，表示購買某知名手搖飲料竟然喝到蟑螂！雖然業者主動表示願意提供該名顧客精神賠償，但蠻好奇的，記得依照法律規定，只有特定情況才可以請求精神賠償。在這種情況下，顧客可以向業者請求精神賠償嗎？",
    },
    {
        "role": "assistant",
        "content": "消費者、業者、精神賠償。"
    }
]

# Initialize pipeline
input_file = "./data/json/test_data.jsonl"
output_file = "./data/json/test_with_keyword.json"
pipe = pipeline("text-generation", model="yentinglin/Taiwan-LLM-7B-v2.0.1-chat", torch_dtype=torch.float16, device_map="auto")

with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as outfile:
    for line in f:
        # Replace 'None' question with title
        item = json.loads(line.strip())
        title = item.get("title")
        question = item.get("question") if item.get("question") and item.get("question") != "None" else item.get("title", "")
        combined_content = f"{title}{question}"

        # Construct messages with few-shot examples and current question
        messages = [
            {
                "role": "system",
                "content": "你是一個熟悉法律的律師助理。請根據以下敘述和問題，生成 3 至 5 個包含法律用語的關鍵字。這些關鍵字應包括常見法律術語、條文名稱，或與描述相關的法律概念。\n",
            }
        ]
        messages.extend(few_shot_examples)
        messages.append({"role": "user", "content": combined_content})

        # Format the input prompt
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate the output
        outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

        # Extract assistant's response
        generated_text = outputs[0]["generated_text"]
        assistant_response = generated_text.split("ASSISTANT")[-1].strip() if "ASSISTANT" in generated_text else generated_text

        # Add assistant response to the 'keyword' field
        item["keyword"] = assistant_response

        # Print for confirmation
        print(f"Question: {combined_content}")
        print(f"Keyword: {assistant_response}\n{'-'*50}")

        outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Updated test_list has been saved to {output_file}")
