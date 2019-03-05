from features import content

datafile = "./data/train.tsv"


labels = []
utterances = []

with open(datafile) as conversations:
	for line in conversations:
		if line != '\n':
			tokens = line.strip().split('\t')
			labels.append(tokens[0])
			utterances.append(tokens[1])
		else:
			# reached the end of the conversation, can extract the features now
			# qm = content.count_qs(text)
			# wh = content.W5H1(text)
			# thanks = content.thanks(text)
			pos = content.abs_position(utterances)
			norm_pos = content.norm_position(utterances)
			


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
						str(norm_pos[i]) + 
						"\n"
					)

			# reset
			labels = []
			utterances = []


		





