from functions import QueryEngine, term_frequency, inverse_doc_freq, tf_idf_matrix, cosine_sim


query = "313 text bbs junkies"



foldername = "data"
engine = QueryEngine(foldername,debug=False, positional=True)
#print(engine.index)
word_count = engine.word_counts
pos_index = engine.index
tf = term_frequency(word_count,pos_index)
idf = inverse_doc_freq(engine.contents,pos_index)

x,y = tf_idf_matrix(tf,idf,query)

print(cosine_sim(x,y))