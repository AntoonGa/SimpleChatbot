## Language Learning Model (LLM) Chatbot
This is a Python-based chatbot that uses OpenAI\'s GPT-4 language model to
    interact with users. The chatbot can be configured with different system
    functions and engines, and it maintains a dynamic chat history that can be
    adjusted as needed.
### Features
- Dynamic chat history that can be flushed, popped, or adjusted to fit within
    token limits. The system also handles its memory itself and flushes out part of the conversation if its context runs too long.
- Customizable system functions and engines
- Token counting and history management
- Streaming API responses with timeout and retry handling
### Requirements
- Python 3.6 or higher
- OpenAI Python library
- Tiktoken library
### Usage
1. Clone the repository and navigate to the project directory.
2. Install the required libraries:
   ```
   pip install openai tiktoken
   ```
3. Set up your OpenAI API key in the `openia_config.txt` file.
4. Run the chatbot:
   ```
   python chatbot.py
   ```
5. Interact with the chatbot by entering queries. You can exit the chat by
    typing "exit", flush the chat history by typing "flush", or remove the oldest
    element in the chat history by typing "pop". You can also change the system
    function by typing "system" or change the engine by typing "engine".
### Example
```
Systems: [\'commenter\', \'coder\', \'chatbot\']
System function: coder
Engines: [\'gpt4\', \'gpt3\']
Engine: gpt4
Action: [\'ask anything:)\', \'exit (stop chat)\', \'flush (destory memory, keep
    system)\', \'pop (remove first question, keep system)\', \'system (destroy
    memory, change system)\', \'engine (change the engine, keep system and
    memory)\']
------------------------
Query:
How to reverse a string in Python?
------------------------
Answer:
You can reverse a string in Python using slicing. Here\'s an example:
```python
string = "Hello, World!"
reversed_string = string[::-1]
print(reversed_string)
```
This will output: `!dlroW ,olleH`
```
### License
This project is licensed under the MIT License.'