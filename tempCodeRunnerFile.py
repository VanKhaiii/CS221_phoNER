import pickle

def load_variable(filename):
    with open(filename, 'rb') as file:
        variable = pickle.load(file)
        file.close()
    return variable
    
word_tokenizer = load_variable("D:\\Nam3\\NLP\\deloy_app\\checkpoint\\bilstm\\word_tokenizer.pkl")
print(word_tokenizer)