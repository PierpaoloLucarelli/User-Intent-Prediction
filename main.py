from features import content
import os

datafile = "./data/train.tsv"


labels = []
utterances = []
user_types = []

with open(datafile) as conversations:
	os.remove("./data/features.csv")
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



			# write features to file
			with open("./data/features.csv", "a") as fout:
				for i, u in enumerate(utterances):

					# label
					# question mark
					# 5W1H
					# thanks
					fout.write(
						labels[i] + ", " +
						str(pos[i]) + ", " +
						str(norm_pos[i]) + ", " +
						str(isUser[i])  + 
						"\n"
					)

			# reset
			labels = []
			utterances = []


		





