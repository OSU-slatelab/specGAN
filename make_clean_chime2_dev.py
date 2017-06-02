

fclean = open("feats_dev_clean.scp")
fnoisy = open("feats_dev_noisy.scp")


fclean_orig = open("feats_dev_clean_orig.scp","w")
dictnoisy={}
dictclean={}
for line in fclean:
	utt_id, path = line.split()
	dictclean[utt_id] = path


for line in fnoisy:
	utt_id, path = line.split()
	dictnoisy[utt_id] = path


for utt_id in dictnoisy.keys():
	search_key = utt_id[:-1]+'0'
	search_path = dictclean[search_key]
	fclean_orig.write(search_key+" "+search_path+"\n")
	

