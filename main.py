from features import content

datafile = "./data/train.tsv"


labels = []
text = []

with open(datafile) as utterances:
	for line in utterances:
		if line != '\n':
			tokens = line.strip().split('\t')
			labels.append(tokens[0])
			text.append(tokens[1])
			#tf-idf for text

		else:
			# reached the end of the conversation, can extract the features now
			
			qm = content.count_qs(text)
			wh = content.W5H1(text)
			thanks = content.thanks(text)


			# write features to file
			with open("./data/features.csv", "a") as fout:
				for i, utterance in enumerate(text):

					# label
					# question mark
					# 5W1H
					# thanks
					fout.write(
							labels[i] + ", " +
							str(qm[i]) + ", " +
							", ".join(wh[i]) + ", " +
							str(thanks[i]) + 
							"\n"
						)

			# reset
			labels = []
			text = []


		





