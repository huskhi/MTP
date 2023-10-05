import streamlit as st
import joblib
import faiss
import numpy as np
import time
import json  # Import the appropriate library for your data format


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

with col1:
    button1 = st.button('FlatL2 Index (Closest Results)')
with col2:
    button2 = st.button('Partitioning the indices')
with col3:
    button3 = st.button("Quantization (Faster)")






# if st.button():
# @st.cache_data()
if button1:

    if user_input:

        start_time = time.time()

        # st.write(index.ntotal)
        D, I = index_L2.search( ip , k)  #
        output = [f'{[i+1] } {sentences[idx]}' for i, idx in enumerate(I[0])]
        print(output)
        for item in output:
            st.write(item)
        end_time = time.time()
        elapsed_time =round( end_time - start_time , 4)
        st.write(f"Time to compute: {elapsed_time} seconds")

    else:
        st.write('Please enter text and k value for Citation.')


if button2:

    if user_input  :
        start_time = time.time()
        # nproval = st.text_input('Enter value of nprobe:', '10')
        # nproval = int(nproval)
        # nlist = st.text_input('partitions in index', '50')
        # nlist = int (nlist)
        quantizer = faiss.IndexFlatL2(d)
        
        # st.write(index.ntotal)
        D, I = index_P.search( ip , k)  #
        output = [f'{[i+1] } {sentences[idx]}' for i, idx in enumerate(I[0])]
        print(output)
        for item in output:
            st.write(item)
        end_time = time.time()
        elapsed_time =round( end_time - start_time , 4)
        st.write(f"Time to compute: {elapsed_time} seconds")
        
    else:
        st.write('Please enter text and k value for Citation.')

if button3:

    if user_input :
        start_time = time.time()
        # m = st.text_input('Number of centroid IDs in final compressed vectors', '8')
        # m = int(m)
        # bits  = st.text_input('number of bits in each centroid', '8')
        # bits = int (bits)
        # nlist = st.text_input('partitions in index', '50')
        # nlist = int (nlist)


        D, I = index_Q.search(ip, k)
        output = [f'{[i+1] } {sentences[idx]}' for i, idx in enumerate(I[0])]
        print(output)
        for item in output:
            st.write(item)
        end_time = time.time()
        elapsed_time =round( end_time - start_time , 4)
        st.write(f"Time to compute: {elapsed_time} seconds")
    else:
        st.write('Please enter text and k value for Citation.')

# Make predictions using the loaded model


