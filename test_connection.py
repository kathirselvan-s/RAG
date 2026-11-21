import ollama

client = ollama.Client(
    host="http://127.0.0.1:11434"
)

try:
    response = client.list()

    print("Ollama connected successfully!")

    for model in response.models:
        print("Model:", model.model)

except Exception as e:
    print("Error:", repr(e))