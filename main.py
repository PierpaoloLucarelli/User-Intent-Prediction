from features import content
import os

datafiles = ["train", "test"]



labels = []
utterances = []
user_types = []

for i, f in enumerate(datafiles):
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
				# qm = content.count_qs(text)
				# wh = content.W5H1(text)
				# thanks = content.thanks(text)
				pos = content.abs_position(utterances)
				norm_pos = content.norm_position(utterances)
				isUser = content.isUser(user_types)
				sentiments = content.vaderSentiment(utterances)
				# words = content.numberOfWords(utterances)


				# write features to file
				with open("./data/"+datafiles[i]+"_feat.csv", "a") as fout:
					for j, u in enumerate(utterances):

						# label
						# question mark
						# 5W1H
						# thanks
						fout.write(
							labels[j] + ", " +
							str(pos[j]) + ", " +
							str(norm_pos[j]) + ", " +
							str(isUser[j])  + ", " + 
							str(sentiments[i]["neu"]) + ", " + 
							str(sentiments[i]["pos"]) +
							# str(sentiments[i]["pos"]) + 
							"\n"
						)

				# reset
				labels = []
				utterances = []


		





