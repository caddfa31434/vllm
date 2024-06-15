
curl http://10.235.103.22:8081/test/test \
  -H "Content-Type: application/json" \
  -XPOST \
  -d '{
    "model": "/workspace/models/agi_llama_pet_65b-release-450-hf",
    "messages": [
      {
        "role": "user",
        "content": "你好，你是谁？"
      }
    ],
    "temperature": 0.2
  }'
