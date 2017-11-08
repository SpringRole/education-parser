from os import listdir
from os.path import isfile,join
import sys
import spacy
import random
from spacy.gold import GoldParse
from spacy.language import EntityRecognizer
#from spacy.language import Doc


def file_names(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]      # A list of filenames in the folder given by PATH
    return onlyfiles


def extract_tags(z):           # training files pattern: TOKEN\tTAG
	words=[]                   
	tags=[]                    # BILOU Encoding scheme used
	f = open(PATH+z,'r')
	for x in f:
		try:
		   word, tag = x.split("\t")
	    	   words.append(word)
	    	   if(tag[0]=="O"):
	    	   	tags.append("O")           
	    	   else:
	    	   	if tag[:-1][-1:]=='\r':              # Because Windows sometimes uses '\r\n' instead of '\n'. I know -_-
	    	   		tags.append((tag[:-2][0]+"-"+tag[:-2][1:]).strip()) # While creating training files '-' wasn't added to reduce some work
	    	   	else:
	    	   		tags.append((tag[:-1][0]+"-"+tag[:-1][1:]).strip())
		except:
	    	   pass
	f.close()
	return words,tags


def train():
	nlp = spacy.load('en')
	nlp.entity.model.learn_rate = 0.001
	nlp.entity.add_label('BRMA')        # BRMA Branch Major
	nlp.entity.add_label('BRMI')        # BRMI Branch Minor
	nlp.entity.add_label('INST')        # INST Institute
	nlp.entity.add_label('DEG')         # DEG Degree
	nlp.entity.add_label('GPE')         # GPE Locations
	nlp.entity.add_label('DATE')        
	nlp.entity.add_label('BRD')         # BRD Board
	nlp.entity.add_label('PERCENT')
	nlp.entity.add_label('GRD')         # GRD Grade
	nlp.entity.add_label('ORDINAL')     # ORDINAL First, Second,etc
	fnames = file_names(PATH)
	
	defected=[]                         # Wrong training files being catched in the except block
	count = 1
	maxloss=0
	for x in range(100):
		random.shuffle(fnames)          # To Train multiple times on same files
		loss = 0.00
		lossupd=0.00
		for z in fnames:
			words, tags = extract_tags(z)
			doc = nlp.make_doc(" ".join(words).decode('utf-8'))
			nlp.tagger(doc)
			#doc = Doc(nlp.vocab, words)
			# print doc
			# print tags
			# print z
			try:
				gold = GoldParse(doc, entities =tags)
			except:
				# defected.append(z)
				continue
			
			loss = nlp.entity.update(doc, gold)
			lossupd += loss
			maxloss=max(maxloss,loss)
			print "loss: ",loss
			print "loss upd",lossupd
			print "Done...",count,z
			count+=1
		if lossupd==0:
			print "Done early",loss,lossupd
			break
		print "MAX---",maxloss
	# print defected, len(defected)
	nlp.end_training()                       # 'nlp' variable now contains the trained model instance

	# print nlp.entity.cfg['extra_labels']
	print "\nTraining Done "
	print "Test Data: "
	print "Rowan University Glassboro, New Jersey Bachelor of Science: Computer Science May 2017 "
	print "\nResults:\n"
	doc = nlp("Rowan University Glassboro, New Jersey Bachelor of Science: Computer Science May 2017 ".decode('utf-8'))
	for ent in doc.ents:
		print ent.label_, ent.text            # CHECKING ON SOME TEST DATA
	nlp.save_to_directory('/home/karan/ResumeParserVer6')     # Model can be then pip installed: https://spacy.io/usage/training#section-saving-loading
                                # Give your own path here


PATH = sys.argv[1]
train()
