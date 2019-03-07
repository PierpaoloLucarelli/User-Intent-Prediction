from features import content
from features import helper
from features import tfidf
import os
import pickle

datafiles = ["train", "test"]



labels = []
utterances = []
user_types = []

for i, f in enumerate(datafiles):
	k = 0
	# tfidf.create_vectorizer('data/'+datafiles[i]+'.tsv', datafiles[i])
	# with open(datafiles[i]+'.pkl','rb') as f:
	tfidf_init = tfidf.calculate_similarities_init(datafiles[i])
	tfidf_dialog = tfidf.calculate_similarities_dialog(datafiles[i])
	with open("./data/"+datafiles[i]+".tsv") as conversations:
		if os.path.exists("./data/"+datafiles[i]+"_feat.csv"):
			os.remove("./data/"+datafiles[i]+"_feat.csv")
		else:
			print("Can not delete the file as it doesn't exists")
		for line in conversations:
			if line != '\n':
				tokens = line.strip().split('\t')
				labels.append(tokens[0])
				utterances.append(tokens[1])
				user_types.append(tokens[2])
			else:
				# reached the end of the conversation, can extract the features now
				qm = content.count_qs(utterances)
				wh = content.W5H1(utterances)
				dups = content.duplicates(utterances)
				thanks = content.thanks(utterances)
				pos = content.abs_position(utterances)
				norm_pos = content.norm_position(utterances)
				isUser = content.isUser(user_types)
				ex = content.exclam_mark(utterances)
				sentiments = content.vaderSentiment(utterances)
				n_words = content.numberOfWords(utterances)
				unique_words = content.numberOfUniqueWords(utterances)
				feedback = content.feedback(utterances)


				# write features to file
				with open("./data/"+datafiles[i]+"_feat.csv", "a") as fout:
					# l = helper.label_to_int(labels)
					for j, u in enumerate(utterances):
						fout.write(
							str(labels[j]) + ", " +
							str(pos[j]) + ", " +
							str(norm_pos[j]) + ", " +
							str(isUser[j])  + ", " + 
							str(sentiments[i]["neu"]) + ", " + 
							str(sentiments[i]["pos"]) + ", " + 
							# str(sentiments[i]["neg"]) + ", " +
							str(qm[i]) + ", " + 
							", ".join(wh[i]) + ", " +
							", ".join(dups[i]) + ", " +
							str(thanks[i]) + ", " + 
							str(n_words[i]) + ", " +
							str(unique_words[i]) + ", " +
							str(tfidf_init[k]) + ", " +
							str(tfidf_dialog[k]) + ", " +
							", ".join(feedback[i]) + ", " +
							str(ex[i]) + 
							"\n"
						)
						k = k + 1
				# reset
				labels = []
				utterances = []



	





