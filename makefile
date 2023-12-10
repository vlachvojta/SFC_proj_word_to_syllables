
zip:
	rm -rf src/__pycache__
	zip 23-xvlach22.zip docs/dokumentace.pdf install.sh requirements.txt \
		src/*.py python-scripts/Dataset_creating.ipynb \
		dataset/ssc_29-06-16/set_*/* dataset/long_words_test.txt \
		models/torch_gru_8hid_250batch_21000epochs.pt models/torch_gru_256hid_2layers_bidirectional_yesbias_250batch_800epochs.pt \	

docu:
	pdflatex docs/main.tex

afterdocu:
	rm -f main.aux
	rm -f main.out
	mv main.pdf docs/dokumentace.pdf


clean:
	# rm 23-xvlach22.zip
