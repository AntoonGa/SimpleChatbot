# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 12:56:21 2023

@author: agarc
"""
# imports 
import time
import traceback
import logging
import tiktoken
import openai

# set path for API config
import os
main_path = './'
keys_path = os.path.join(main_path, 'openia_config.txt')

# Set API keys for cvRanker and chunkNorris
keys_file = open(keys_path)
lines = keys_file.readlines()
keys_file.close()
# os.environ["OPENAI_API_TYPE"] = lines[0].split("=")[1].strip()
# os.environ["OPENAI_API_BASE"] = lines[1].split("=")[1].strip()
# os.environ["OPENAI_API_VERSION"] = lines[2].split("=")[1].strip()
# os.environ["OPENAI_API_KEY"] = lines[3].split("=")[1].strip()
openai.api_type = lines[0].split("=")[1].strip()
openai.api_base = lines[1].split("=")[1].strip()
openai.api_version = lines[2].split("=")[1].strip()
openai.api_key = lines[3].split("=")[1].strip()


class llm():
    """
    A copilot chatbot. History is dynamic (pops when too long, can be deleted)
    """

    def __init__(self, llm_engine="gpt-4-32k"):
        """
        Initialize the Chatbot instance with the specified language model engine and token limits.

        This function sets the system function, engine model, and token limits based on the provided engine name.
        It also calculates the maximum token context to avoid overfeeding tokens.

        Args:
            self: The Chatbot instance.
            llm_engine (str, optional): The name of the language model engine to use. Defaults to "gpt-4-32k".

        Returns:
            None
        """
        
        # initiate system function
        self._set_system_function()
        # set system model and context length according to chosen model
        self._set_engine()
        return
    
    def _set_engine(self):
        """
        Set the engine and its corresponding token limits based on the selected engine.
        
        This function sets the engine, maximum tokens allowed in the context, and maximum tokens allowed in the response
        based on the selected engine. It also calculates the maximum token context by subtracting the response tokens
        and a buffer token value from the total maximum tokens.
        
        Args:
            self: An instance of the class containing the engine and token attributes.
            
        Returns:
            None
        """
        
        self.engine = self._get_engine()
            
        if self.engine == "gpt-4-32k":  # this model is rather expensive !
            self.max_tokens = 30000  # maximum token in llm context
            self.max_tokens_in_response = 4000  # maximum token in llm response
        elif self.engine == "gpt-35-turbo":
            self.max_tokens = 8000
            self.max_tokens_in_response = 2000  # maximum token in llm response
        else:
            self.max_tokens = 4000
            self.max_tokens_in_response = 700

        buffer_token = 1000  # avoid over feeding tokens, note that the prompt template needs some tokens !
        self.max_token_context = self.max_tokens - self.max_tokens_in_response - buffer_token  # rest
        return
    
    def _get_engine(self):
        """
        Prompt the user to choose an engine and return the corresponding description.

        This function presents the user with a list of preset engines and allows them to input their choice.
        Based on the user's input, it returns the appropriate engines description.

        Args:
            self: The Chatbot instance.

        Returns:
            str: The description of the chosen system function.
        """

        # add to this dictionnaries different engine that the system can take:)
        hardcoded_engines = {"gpt4": "gpt-4-32k",
                             "gpt3": "gpt-35-turbo"}
        # list roles priot to user input
        print("Engines:", list(hardcoded_engines.keys()))
        # get user inputs
        user_input = input("Engine: ")

        # returns the system functions
        if user_input in hardcoded_engines:
            engine = hardcoded_engines[user_input]
        elif user_input in [None, ""]:
            print("Default Engine : GPT4")
            engine = hardcoded_engines["gpt4"]
        else:
            engine = user_input

        print(engine)
        return engine
    
            
        
        

    def _set_system_function(self):
        """
        Initialize the chatbot's system function and set it as the first entry in the chat history.

        This function retrieves the system function using the _get_system_function method, initializes the role_system
        attribute with the system function, and sets the chat history with the system function as the first entry.

        Args:
            self: The Chatbot instance.

        Returns:
            None
        """
        # get system function
        system_function = self._get_system_function()
        # init the system on call
        self.role_system = {'role': 'system', 'content': system_function}
        # history counter
        self.history = [self.role_system]
        return

    def _get_system_function(self):
        """
        Prompt the user to choose a system function and return the corresponding description.

        This function presents the user with a list of preset system functions and allows them to input their choice.
        Based on the user's input, it returns the appropriate system function description.

        Args:
            self: The Chatbot instance.

        Returns:
            str: The description of the chosen system function.
        """

        # add to this dictionnaries different roles that the system can take:)
        hardcoded_systems = {"commenter": """
        I will provide python functions. 
        You will provide description headers for these functions. Be concisce. 
        First interpret what the function do. Then write a short description of the function. 
        Next in the header write what the arguement and outputs are in the pythonic triple quote format.
        Also add comment to lines of code that are complicated or use external libraries
        """,
                             "coder": """
        You are a coding assistant for Python developpers. A Python co-pilot !
        You are consice, precice and code at the highest level.
        You use a wide variety of famous Python package and library.
        You provide codes with good comments and functions headers indicating the types of output and arguments.
        When you provide code, make sur it is well delimited from your other sentenses.
        """,
                             "chatbot": """
        You are a helpfull assistant.
        You provide accurate answer to users queries.
        """}
        # list roles priot to user input
        print("Systems:", list(hardcoded_systems.keys()))
        # get user inputs
        user_input = input("System function: ")

        # returns the system functions
        if user_input in hardcoded_systems:
            system_function = hardcoded_systems[user_input]
        elif user_input in [None, ""]:
            system_function = hardcoded_systems["coder"]
        else:
            system_function = user_input

        print(system_function)
        return system_function

    def _count_tokens(self, encoding_name="cl100k_base"):
        """
        Calculate the total number of tokens in the chat history.
        Generates a full string from the chat history
        Computes the number of tokens using the specified encoding model, and adds the number of keys from the history to the context length.

        Args:
            encoding_name (str, optional): The name of the encoding model to use. Defaults to "cl100k_base".

        Returns:
            int: The total number of tokens in the chat history.
        """
        # generate full string from history
        string_total = ''.join([chat["content"] for chat in self.history])
        # compute number of tokens tokens
        encoding = tiktoken.encoding_for_model("gpt-4")
        num_tokens = len(encoding.encode(string_total))
        # add the keys from history in the context length
        num_tokens += len(self.history)
        return num_tokens

    def _adjust_history_size(self):
        """
        Adjust the chat history size to ensure it does not exceed the maximum token context.

        This function calculates the number of tokens in the chat history and reduces the context size if needed by
        removing the oldest elements until the history size is within the allowed limit.
        """
        # assign first number to enter the while loop
        number_of_tokens_in_history = self._count_tokens()
        # reduce context if needed
        if number_of_tokens_in_history >= self.max_token_context:
            # in a while loop, until the history is of appropriate size
            while number_of_tokens_in_history >= self.max_token_context:
                print("\n ----- max context size reached. Poping. ----- \n")
                # remove the element after the system message (oldest element)
                self._pop_history()
                # count again
                number_of_tokens_in_history = self._count_tokens()
        return

    def _flush_history(self):
        """
        Flush the chat history, keeping only the initial system message.
        This function removes all elements from the chat history except the first one, which is the initial system message.
        """
        self.history = [self.history[0]]
        return

    def _pop_history(self):
        """
        Remove the oldest question and answer from the chat history.
        """
        try:
            self.history.pop(1)
            self.history.pop(1)
        except Exception as e:
            logging.error('Failed to remove second element of history: ' + str(e))
        return

    def _append_history(self, role, content):
        """
        Append a new message to the chat history.
        This function adds a new message to the chat history with the specified role and content.

        Args:
            role (str): The role of the message sender, e.g., "user" or "assistant".
            content (str): The content of the message.

        Returns:
            None
        """
        # place new user message in history
        new_message = {"role": role, "content": content}
        self.history.append(new_message)
        return

    def _send_payload_stream_answer(self, payload):
        """
        Send the payload to the OpenAI API and stream the response, handling timeouts and retries.

        This function sends the payload to the OpenAI API using the ChatCompletion.create method with streaming enabled.
        It handles timeouts and retries up to 3 times before returning the response.

        Args:
            self: The Chatbot instance.
            payload (dict): The payload to send to the OpenAI API.

        Returns:
            str: The response from the OpenAI API.
        """
        response = ''
        # we make multiple tries with a 10second timeout
        is_to_do = 1  # break condition 1
        try_counter = 0  # break contition 2: will break if number of attempts is 5
        while is_to_do == 1 and try_counter <= 3:
            try:
                for chunk in openai.ChatCompletion.create(
                        engine=self.engine,
                        temperature=0,
                        max_tokens=self.max_tokens_in_response,
                        messages=payload,
                        stream=True,
                        timeout=10):
                    content = chunk["choices"][0].get("delta", {}).get("content")
                    if content is not None:
                        print(content, end='')
                        response = response + content

                is_to_do = 0

            except Exception as e:
                logging.error(traceback.format_exc())
                print("Timeout. Retry:", try_counter)
                try_counter += 1
                time.sleep(1)

        return response

    def _send_receive_message(self, query):
        """
        Send a single message without using history as payload and append the response to the history.

        This function sends the full history as payload, receives the response from the language model, and appends
        the response to the chat history. The history is used for debugging and monitoring purposes.

        NOTE: as of now this function does nothing but call self._send_payload_stream_answer(payload)
        Other features will be stacked in this function, so I keep it.

        Args:
            self: The Chatbot instance.
            query (str): The message to send to the language model.

        Returns:
            str: The response from the language model.
        """
        # send full history has payload
        payload = self.history
        # send history to llm and get response
        response = self._send_payload_stream_answer(payload)
        return response

    def chat(self):
        """
        Interact with the chatbot in a continuous loop.

        This function allows the user to interact with the chatbot by entering queries. The user can exit the chat by
        typing "exit", flush the chat history by typing "flush", or remove the oldest element in the chat history by
        typing "pop". For other queries, the function appends the user query to the history, adjusts the history size if
        needed, sends the query to the chatbot, receives the response, and appends the response to the history.

        Args:
            None

        Returns:
            None
        """
        action_list = ["ask anything:)",
                       "exit (stop chat)", 
                       "flush (destory memory, keep system)", 
                       "pop (remove first question, keep system)", 
                       "system (destroy memory, change system)",
                       "engine (change the engine, keep system and memory)"]

        while True:
            time.sleep(0.5)
            print('\n______________________________________________________')
            print('\n______________________________________________________')
            print("Action:", action_list)
            print("------------------------")
            user_query = input("Query:\n")

            if user_query:
                # kills the process
                if user_query == "exit":
                    break

                # destroy history, keep system function
                elif user_query == "flush":
                    self._flush_history()

                # pop oldest element of history, keep system function
                elif user_query == 'pop':
                    self._pop_history()

                # destroy history, reset system function:
                elif user_query == "system":
                    self._set_system_function()
                    
                elif user_query == "engine":
                    self._set_engine()

                else:
                    print("------------------------")
                    print('Answer: \n')
                    # append history with user query
                    self._append_history(role="user", content=user_query)
                    # remove part of history (first after system) if history is too long
                    self._adjust_history_size()
                    # send and receive message
                    response = self._send_receive_message(user_query)
                    # append history with system message
                    self._append_history(role="assistant", content=response)

        return


# %%

def main():
    chatbot = llm(llm_engine="gpt-4-32k")
    chatbot.chat()

if __name__ == "__main__":
    main()
    

    
