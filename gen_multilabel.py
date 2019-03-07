with open("labels2.csv") as file:
	labelList = []
	for line in file:
		labelList = line.strip().split(',')

	datafiles = ["train", "test"]
	for i, f in enumerate(datafiles):
		labels = []
		with open("./data/"+datafiles[i]+".tsv") as conversations:
			for line in conversations:
				tokens = line.strip().split('\t')
				if(tokens[0] != ""):
					labels.append(tokens[0])
			vectorizedLabels = [[False]*len(labelList)]*len(labels)
			for label in labels:
				for 






