import streamlit as st
import joblib
import faiss
import numpy as np
import time
import re
import requests
import streamlit as st
from transformers import BertTokenizer, BertModel
import torch

def load_model():
    return joblib.load('cite_model.pkl')
model = load_model()
# Load your trained model from the .pkl file
model = joblib.load('cite_model.pkl')
st.title('Cite App')

# Add input widgets (e.g., text input, slider, file upload) as needed
user_input = st.text_input('Enter text for Citation:', '')
k = st.text_input('Enter number of citations required:', '3')
k = int(k)
ip = model.encode([user_input])

st.session_state['button'] = False
## code to load and combine embeddings
# emb1 = np.load('embeddings_0.npy' )
# emb2= np.load('embeddings_1.npy' )
# emb3 = np.load('embeddings_2.npy' )
# emb4  = np.load('embeddings_3.npy' )
# combined_file_path = 'combined_embeddings.npy'

# # Vertically stack the embeddings
# combined_embeddings = np.vstack((emb1, emb2, emb3 , emb4))
# np.save( 'combined_embeddings.npy', combined_embeddings)

#load embeddings
sentence_embeddings = np.load('embeddings1k_title.npy')

#read the sentences 
with open('sentences1k_title.txt' ,  'r') as file:
    # Read the entire content of the file into a string variable
    sentences = file.read()
sentences = sentences.splitlines()
d = sentence_embeddings.shape[1]
st.write (sentence_embeddings.shape[0])
       
m = 8  # number of centroid IDs in final compressed vectors
bits = 8 # number of bits in each centroid
nlist = 50
quantizer = faiss.IndexFlatL2(d)  # we keep the same L2 distance flat index
index_Q = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits) 
index_Q.train(sentence_embeddings)
index_Q.add(sentence_embeddings)
index_Q.nprobe = 10

index_L2 = faiss.IndexFlatL2(d)
index_L2.add(sentence_embeddings) 

nproval = 10
index_P = faiss.IndexIVFFlat(quantizer, d, nlist)
index_P.train(sentence_embeddings)
index_P.add(sentence_embeddings)
index_P.nprobe = nproval
index_P.make_direct_map()

col1, col2, col3 = st.columns([1,1,1])
regen = ""

with col1:
    button1 = st.button('FlatL2 Index (Closest Results)')
with col2:
    button2 = st.button('Partitioning the indices')
with col3:
    button3 = st.button("Quantization (Faster)")

######################### ATTENTION #############################
def calculate_top_words_with_attention(input_text):
    common_words = [
    'a', 'an', 'the', 'and', 'in', 'on', 'at', 'of', 'for', 'to',
    'with', 'by', 'as', 'is', 'are', 'was', 'were', 'it', 'that',
    'this', 'these', 'those', 'there', 'here', 'from', 'or', 'but',
    'not', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'can',
    'could', 'shall', 'should', 'will', 'would', 'may', 'might',
    'must', 'being', 'been', 'am', 'are', 'is', 'was', 'another',
    'while', 'during', 'after', 'before', 'under', 'over', 'above',
    'below', 'between', 'among', 'beside', 'along', 'against', 'upon',
    'within', 'without', 'through', 'throughout', 'inside', 'outside',
    'beyond', 'against', 'once', 'twice', 'thrice', 'such', 'every',
    'each', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
    'eight', 'nine', 'ten', 'eleven', 'twelve', 'first', 'second',
    'third', 'last', 'few', 'several', 'many', 'much', 'more', 'most',
    'less', 'least', 'half', 'whole', 'part', 'piece', 'bit', 'while',
    'where', 'when', 'why', 'how', 'who', 'whom', 'whose', 'which',
    'what', 'whether', 'while', 'whenever' , 'however',
    'whoever', 'whichever', 'whatever', 'before', 'after', 'during',
    'since', 'until', 'ago', 'later', 'early', 'soon', 'now', 'then',
    'today', 'tonight', 'tomorrow', 'yesterday', 'forever', 'always',
    'never', 'sometimes', 'often', 'rarely', 'seldom', 'usually',
    'frequently', 'occasionally', 'daily', 'weekly', 'monthly',
    'annually', 'yearly', 'sometimes', 'often', 'never', 'rarely',
    'always', 'usually', 'generally', 'typically', 'mostly', 'largely',
    'primarily', 'chiefly', 'mainly', 'overall', 'overall', 'overall',
    'indeed', 'indeed', 'indeed', 'however','hence', 'hence', 'hence', 'thus',   'therefore',
    'consequently', 'moreover',  'furthermore','additionally', 'besides',  'alternatively', 'nevertheless', 
    'however', 'otherwise','regardless', 'meanwhile', 'subsequently'
]

    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)

    # Tokenize the text
    tokens = tokenizer.encode_plus(input_text, return_tensors='pt', padding=True, truncation=True)

    # Get model attentions
    with torch.no_grad():
        outputs = model(**tokens)
        attentions = torch.stack(outputs.attentions)  # Stack attention tensors

    # Calculate average attention weights across all layers
    avg_attentions = torch.mean(attentions, dim=0)  # Calculate mean along the layers dimension

    # Find the words with the highest average attention
    avg_attentions_per_token = torch.mean(avg_attentions, dim=0)  # Calculate mean along the heads dimension

    # Convert token IDs back to words
    decoded_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    word_attentions = list(zip(decoded_tokens, avg_attentions_per_token.tolist()))
    word_attentions = [(word, attention) for word, attention in word_attentions if word not in common_words]
    # Sort words based on average attention
    word_attentions.sort(key=lambda x: x[1], reverse=True)

    return word_attentions

# Streamlit app


# if st.button('Calculate'):
def highlight_para(input_text):
    # Calculate top words with attention
    top_words = calculate_top_words_with_attention(input_text)
    # top_words = [word for word, _ in word_attentions if word not in common_words]
    # st.write(top_words)
    # Prepare output as a string for Streamlit
    output = "Top Words with Highest Average Attention:\n"

    tokenized_text = input_text.split()  # Tokenize the input text
    highlighted_text = ""

    top_word_set = {word for word, _ in top_words}  # Set of top words for efficient lookup
    for token in tokenized_text:
        if token in top_word_set:
            highlighted_text += f"<span style='background-color: yellow;'>{token}</span> "
        else:
            highlighted_text += f"{token} "

    # st.write(top_words)
    st.markdown(highlighted_text, unsafe_allow_html=True)  # Display highlighted text

################################



##############
selected_indices = []


    
from sklearn.metrics.pairwise import cosine_similarity
if button1:

    if user_input:
        st.session_state['button'] = True
        highlight_para(user_input)

        start_time = time.time()

        # st.write(index.ntotal)
        D, I = index_L2.search( ip , k)  #
        output = [f' {sentences[idx]}' for i, idx in enumerate(I[0])]
     
        ip = np.squeeze(ip)  # Remove single-dimensional entries from the shape of an array
        ip = np.atleast_2d(ip)  # Ensure ip is a 2D array

        sentence_embeddings_I = np.squeeze(sentence_embeddings[I])  # Remove single-dimensional entries and convert to 2D array

        # Compute cosine similarity
        similarity_scores = cosine_similarity(ip, sentence_embeddings_I)

        similarity_scores = similarity_scores[0]
        similarity_scores = list(similarity_scores)
 
        for i, item in enumerate(output):
            
            st.markdown(f"<div style='background-color: lightgreen; padding: 10px;'><b>Citation score:</b> {similarity_scores[i]:.3f}</div>", unsafe_allow_html=True)
            if len(item) > 10:
                with  st.expander(f"[{i+1}] {item[0:50]}"  ):
                    highlight_para(item)
                    
                    regen += item
            else:
                with st.expander(item[0:50]):
                    highlight_para(item)
                    regen+=item

        end_time = time.time()
        st.session_state['text'] = regen
        elapsed_time =round( end_time - start_time , 4)
        st.write(f"Time to compute: {elapsed_time} seconds")

    else:
        st.write('Please enter text and k value for Citation .')



if button2:

    if user_input  :
        st.session_state['button'] = True
        start_time = time.time()
        # nproval = st.text_input('Enter value of nprobe:', '10')
        # nproval = int(nproval)
        # nlist = st.text_input('partitions in index', '50')
        # nlist = int (nlist)
        quantizer = faiss.IndexFlatL2(d)
        
        # st.write(index.ntotal)
        
        D, I = index_P.search( ip , k)  #
        output = [f' {sentences[idx]}' for i, idx in enumerate(I[0])]


        ip = np.squeeze(ip)  # Remove single-dimensional entries from the shape of an array
        ip = np.atleast_2d(ip)  # Ensure ip is a 2D array

        sentence_embeddings_I = np.squeeze(sentence_embeddings[I])  # Remove single-dimensional entries and convert to 2D array

        # Compute cosine similarity
        highlight_para(user_input)
        similarity_scores = cosine_similarity(ip, sentence_embeddings_I)

        similarity_scores = similarity_scores[0]
        similarity_scores = list(similarity_scores)
 
        for i, item in enumerate(output):
            
            st.markdown(f"<div style='background-color: lightgreen; padding: 10px;'><b>Citation score:</b> {similarity_scores[i]:.3f}</div>", unsafe_allow_html=True)
            if len(item) > 10:
                with  st.expander(f"[{i+1}] {item[0:50]}"  ):
                    highlight_para(item)
                    
                    regen += item
            else:
                with st.expander(item[0:50]):
                    highlight_para(item)
                    regen+=item


if button3:

    if user_input :
        st.session_state['button'] = True
        start_time = time.time()
        # m = st.text_input('Number of centroid IDs in final compressed vectors', '8')
        # m = int(m)
        # bits  = st.text_input('number of bits in each centroid', '8')
        # bits = int (bits)
        # nlist = st.text_input('partitions in index', '50')
        # nlist = int (nlist)


        D, I = index_Q.search(ip, k)
        output = [f' {sentences[idx]}' for i, idx in enumerate(I[0])]


        ip = np.squeeze(ip)  # Remove single-dimensional entries from the shape of an array
        ip = np.atleast_2d(ip)  # Ensure ip is a 2D array

        sentence_embeddings_I = np.squeeze(sentence_embeddings[I])  # Remove single-dimensional entries and convert to 2D array

        # Compute cosine similarity
        highlight_para(user_input)
        similarity_scores = cosine_similarity(ip, sentence_embeddings_I)

        similarity_scores = similarity_scores[0]
        similarity_scores = list(similarity_scores)
 
        for i, item in enumerate(output):
            
            st.markdown(f"<div style='background-color: lightgreen; padding: 10px;'><b>Citation score:</b> {similarity_scores[i]:.3f}</div>", unsafe_allow_html=True)
            if len(item) > 10:
                with  st.expander(f"[{i+1}] {item[0:50]}"  ):
                    highlight_para(item)
                    
                    regen += item
            else:
                with st.expander(item[0:50]):
                    highlight_para(item)
                    regen+=item


# if st.session_state['button']:
#     highlight_para(user_input)




############ Regenerate Text ################

st.session_state['regen'] = regen
if st.session_state['button']:
    if st.button('Regenerate'):
        regen = ""
        output = st.session_state['text']
        output = re.split(r'\[\d+\]', output)
        for i, item in enumerate(output):
            
                    if len(item) > 50:
                        with st.expander(item[0:50]):
                            st.write(item)
                            regen+= item
                            
                    else:
                        with st.expander(item[0:50]):
                            st.write(item)
                            regen+= item
        st.write(regen)
        API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
        headers = {"Authorization": "Bearer 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"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
            
        regen = regen[0:30]
        output = query({
            "inputs": f" Write the given text to make it more technical  - {regen} ",
        })

        st.write(output)
        regen = output


########################################################

# from transformers import BertTokenizer, BertModel
# import torch

# # Load pre-trained BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# # Function to compute similarity and access attention weights
# def compute_similarity_and_attention(paragraph_1, paragraph_2):
#     # Tokenize paragraphs
#     inputs = tokenizer(paragraph_1, paragraph_2, return_tensors='pt', padding=True, truncation=True)
    
#     # Get model predictions for the tokenized inputs
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # Extract the last layer hidden states from the model output
#     hidden_states = outputs.last_hidden_state

#     # Calculate cosine similarity between the hidden states of the two paragraphs
#     similarity_scores = torch.nn.functional.cosine_similarity(hidden_states[:, 0, :], hidden_states[:, 1, :], dim=1)
    
#     # Attention weights can be accessed directly from the output
#     attention_weights = outputs.attentions
    
#     return similarity_scores.item(), attention_weights

# # Streamlit app layout and functionality
# st.title("Paragraph Similarity Checker")

# # Input paragraphs
# paragraph_1 = st.text_area("Enter the first paragraph:")
# paragraph_2 = st.text_area("Enter the second paragraph:")

# # Button to compute similarity and visualize attention
# if st.button("Calculate Similarity"):
#     if paragraph_1 and paragraph_2:
#         similarity_score, attention_weights = compute_similarity_and_attention(paragraph_1, paragraph_2)
#         st.write(f"Similarity Score: {similarity_score}")
#         # Display attention weights if needed
#         st.write("Attention Weights:")
#         st.write(attention_weights)  # Visualize attention weights
#     else:
#         st.warning("Please enter both paragraphs to calculate similarity.")


# #############

# # Paragraph content
# paragraphs = [
#     "This is the first paragraph.",
#     "This is the second paragraph.",
#     "This is the third paragraph."
# ]

# # Checkbox for each paragraph


# # Regenerate button






# # Make predictions using the loaded model


